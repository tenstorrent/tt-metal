# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only native regression of the production prewarm failure policy.

Compiles the real host-only policy header. No device, weights, or Metal library is loaded.
The integration harness additionally compiles run_prewarm directly from its production source,
with controlled build outcomes and lightweight executor/manifest dependencies.
"""

from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]

NATIVE = r'''
#include "tt_metal/impl/program/prewarm_failure.hpp"
#include <cassert>
#include <iostream>
#include <thread>
using namespace tt::tt_metal::kernel_prewarm;
namespace fs = std::filesystem;

int main(int, char** argv) {
    fs::path root(argv[1]);
    fs::create_directories(root);
    auto bad = failure::command({"/bin/sh", "-c", "echo 'x.cpp: error: static assertion failed: Index out of range' >&2; exit 1"}, root);
    assert(bad.definite_compile_failure());
    int attempts = 0;
    std::string context = "source-a-header-a-toolchain-a-recipe-a";
    auto fingerprint = [&]() { return std::optional<std::string>(context); };
    auto build = [&]() { ++attempts; throw failure::CompileFailure(bad, true); };
    auto cleanup = []() {};
    auto state = root / "failed";
    auto run = [&](bool retry = false) { return failure::target(state, retry, fingerprint, build, cleanup); };
    assert(run().outcome == failure::Outcome::Failed);
    assert(run().outcome == failure::Outcome::PreviouslyFailed && attempts == 1);
    // Each identity dimension must invalidate without changing build_key.
    for (const char* changed : {"source-b", "header-b", "toolchain-b", "recipe-b"}) {
        context = changed;
        assert(run().outcome == failure::Outcome::Failed);
        assert(run().outcome == failure::Outcome::PreviouslyFailed);
    }
    assert(run(true).outcome == failure::Outcome::Failed && attempts == 6);
    fs::remove(state); // documented explicit reset
    assert(run().outcome == failure::Outcome::Failed && attempts == 7);
    auto success = failure::target(state, true, fingerprint, []() {}, cleanup);
    assert(success.outcome == failure::Outcome::Successful && !fs::exists(state));
    // Missing/truncated/extra state never suppresses work.
    for (const char* corrupt : {"", "prewarm-failure-v1", "prewarm-failure-v1 recipe-b\ntrailing"}) {
        std::ofstream(state) << corrupt;
        assert(run().outcome == failure::Outcome::Failed);
    }
    // Unknown context never caches a failure, including an existing matching marker.
    auto unknown = []() { return std::optional<std::string>{}; };
    assert(failure::target(state, false, unknown, build, cleanup).outcome == failure::Outcome::Failed);
    assert(!fs::exists(state));
    // A context changed during compilation must not be recorded.
    auto changes = [&]() { context += "changed"; throw failure::CompileFailure(bad, true); };
    assert(failure::target(state, false, fingerprint, changes, cleanup).outcome == failure::Outcome::Failed);
    assert(!fs::exists(state));
    // Normal nonzero statuses, missing executable, signal, timeout and infrastructure messages
    // all remain retryable. Link failures are not negatively cached either.
    for (auto command : std::vector<std::vector<std::string>>{
        {"/bin/sh", "-c", "echo 'x: error: bad'; exit 42"},
        {"/no/such/compiler"},
        {"/bin/sh", "-c", "kill -TERM $$"},
        {"/bin/sh", "-c", "echo 'x: error: bad'; echo 'fatal error: Killed signal'; exit 1"},
        {"/bin/sh", "-c", "echo 'x: error: bad'; echo 'No space left on device'; exit 1"}}) {
        auto result = failure::command(command, root);
        assert(!result.success() && !result.definite_compile_failure());
        auto transient = [&]() { throw failure::CompileFailure(result, true); };
        for (int repeat = 0; repeat != 2; ++repeat) {
            assert(failure::target(state, false, fingerprint, transient, cleanup).outcome == failure::Outcome::Failed);
            assert(!fs::exists(state));
        }
    }
    auto timeout = failure::command({"/bin/sh", "-c", "sleep 30"}, root, std::chrono::milliseconds(50));
    assert(timeout.timed_out && !timeout.success());
    auto link = [&]() { throw failure::CompileFailure(bad, false); };
    failure::target(state, false, fingerprint, link, cleanup);
    assert(!fs::exists(state));
    // An interrupted writer's unfinished temporary record cannot suppress the next pass.
    std::ofstream(state.string() + ".unfinished") << "prewarm-failure-v1 " << context;
    assert(run().outcome == failure::Outcome::Failed);
    fs::remove(state);
    // State publication failures are reported and never turn a failed compile into success.
    auto unavailable = failure::target(root / "missing/state", false, fingerprint, build, cleanup);
    assert(unavailable.outcome == failure::Outcome::Failed);
    assert(unavailable.diagnostic.find("state not saved") != std::string::npos);
    // Concurrent passes serialize an entire kernel, including generated files and target state.
    {
        failure::KernelLock first(root / "kernel.lock");
        failure::KernelLock second(root / "kernel.lock");
        assert(first.acquired() && !second.acquired());
        pid_t pid = fork();
        if (pid == 0) {
            failure::KernelLock other(root / "kernel.lock");
            _exit(other.acquired() ? 1 : 0);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    failure::KernelLock next(root / "kernel.lock");
    assert(next.acquired());
    // Readers see only old or new records, never a partial rename publication.
    failure::save(state, "old");
    std::atomic<bool> writing{true};
    std::thread writer([&]() { for (int n = 0; n < 100; ++n) failure::save(state, n % 2 ? "old" : "new"); writing = false; });
    while (writing) {
        std::ifstream f(state); std::string line; std::getline(f, line);
        assert(line == "prewarm-failure-v1 old" || line == "prewarm-failure-v1 new");
    }
    writer.join();
    // Drain 32 MiB combined stdout/stderr without a spool file or growing buffer.
    auto noisy = failure::command({"python3", "-c",
        "import os; os.write(1,b'HEAD\\n'); [(os.write(1,b'x'*4096),os.write(2,b'y'*4096)) for _ in range(4096)]; os.write(2,b'\\nTAIL\\n')"}, root);
    assert(noisy.success() && noisy.bytes == 33554443);
    assert(noisy.head.size() == failure::diagnostic_half && noisy.tail.size() == failure::diagnostic_half);
    assert(noisy.diagnostic().starts_with("HEAD") && noisy.diagnostic().ends_with("TAIL\n"));
    // Infrastructure text hidden in discarded middle output still prevents negative caching.
    failure::CommandResult middle;
    middle.consume("x: error: bad\n" + std::string(20000, 'a'));
    middle.consume("No space left on device");
    middle.consume(std::string(20000, 'z'));
    middle.exit_code = 1;
    assert(!middle.definite_compile_failure());
    std::cout << "policy: retry, invalidation, reset, success, interruption, atomic state, concurrency passed; "
              << noisy.bytes << " output bytes retained in " << noisy.head.size() + noisy.tail.size() << " bytes\n";
}
'''



# Only external dependencies are substituted: hash backend, JIT cache operations and executor.
# The production context computation, compiler invocation and run_prewarm loop are compiled intact.
INTEGRATION_PREFIX = r"""
#include "tt_metal/impl/program/prewarm_failure.hpp"
#include "tt_metal/jit_build/types.hpp"
#include <atomic>
#include <cassert>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
namespace fs=std::filesystem;
namespace tt {
constexpr int LogMetal=0, LogBuildKernels=0;
// Test backend only; production uses existing BLAKE3 StableHasher.
struct StableHasher {
    uint64_t value=14695981039346656037ULL;
    void update(const void* p,size_t n){auto b=static_cast<const unsigned char*>(p); while(n--){value^=*b++;value*=1099511628211ULL;}}
    void update(uint64_t n){update(&n,sizeof(n));}
    void update(std::string_view s){update(s.data(),s.size());}
    uint64_t digest()const{return value;}
};
}
namespace jit_build=tt::jit_build;
namespace jit_server { struct CompileRequest {
 uint64_t build_key=7; std::string kernel_name,gpp;
 std::vector<jit_build::TargetRecipe> targets;
 std::vector<jit_build::GeneratedFile> generated_files;
}; }
std::vector<size_t> summary;
template<class... Args> void log_warning(int,const char*,Args&&...) {}
template<class... Args> void log_error(int,const char*,Args&&...) {}
template<class... Args> void log_info(int,const char* text,Args&&... args) {
 if(std::string_view(text).starts_with("kernel prewarm: attempted=")) summary={static_cast<size_t>(args)...};
}
namespace tt::jit_build {
using ParsedDependencies=std::unordered_map<std::string,std::vector<std::string>>;
namespace utils {
std::vector<std::string> tokenize_flags(const std::string& s) {
 std::istringstream in(s); std::vector<std::string> out; std::string word; while(in>>word)out.push_back(word); return out;
}
std::string compiler_version(const std::string&) { return "test-compiler-version"; }
}
}
using namespace tt::tt_metal::kernel_prewarm;
const char* env_or_null(const char* name){const char* v=getenv(name);return v && *v ? v:nullptr;}
bool env_truthy(const char* name){const char* v=env_or_null(name);return v && (std::string(v)=="1" || std::string(v)=="true" || std::string(v)=="on");}
void append_tokenized(std::vector<std::string>& args,const std::string& flags){auto words=jit_build::utils::tokenize_flags(flags);args.insert(args.end(),words.begin(),words.end());}
std::unordered_map<std::string,std::string> build_firmware_filename_map(const std::string&){return {};}
std::string base_kernel_name(const std::string& s){return s;}
bool is_device_init_kernel(const std::string&){return false;}
void write_generated_files(const std::string& dir,const std::vector<jit_build::GeneratedFile>& files){fs::create_directories(dir);for(const auto& f:files)std::ofstream(dir+f.name).write(reinterpret_cast<const char*>(f.content.data()),f.content.size());}
std::atomic<int> attempts{0};
void build_target(const std::string& gpp,const jit_build::TargetRecipe& t,const std::string& dir){
 ++attempts; fs::create_directories(dir);
 auto args=jit_build::utils::tokenize_flags(gpp); append_tokenized(args,t.cflags);append_tokenized(args,t.includes);
 args.insert(args.end(),t.defines.begin(),t.defines.end());args.insert(args.end(),{"-c",t.srcs.at(0),"-o",dir+"target.o"});
 auto result=failure::command(args,dir);if(!result.success())throw failure::CompileFailure(result,true);
}
template<class F> void launch_build_step(F f,std::vector<std::shared_future<void>>& events){events.push_back(std::async(std::launch::async,f).share());}
void sync_build_steps(std::vector<std::shared_future<void>>& events){for(auto& f:events)f.get();}
"""

INTEGRATION_MAIN = r"""
int main(int,char** argv) {
 fs::path root=argv[1];fs::create_directories(root/"sfpi/compiler/bin");
 fs::path compiler=root/"sfpi/compiler/bin/g++";
 std::ofstream(compiler)<<"#!/bin/sh\nexec /usr/bin/c++ \"$@\"\n";
 fs::permissions(compiler,fs::perms::owner_all);
 unsetenv("LD_LIBRARY_PATH");unsetenv("LD_PRELOAD");unsetenv("GCC_EXEC_PREFIX");unsetenv("COMPILER_PATH");
 fs::path src=root/"source.cpp", header=root/"header.h", out=root/"cache";
 std::ofstream(src)<<"#include <header.h>\nstatic_assert(VALUE == 2, \"Index out of range\");\n";
 std::ofstream(header)<<"#define VALUE 1\n";
 jit_build::TargetRecipe target;target.target_name="target";target.compiler_opt_level="O0";
 target.srcs={src};target.objs={"target.o"};target.includes="-I"+root.string();
 jit_server::CompileRequest req;req.kernel_name="kernel";req.gpp=compiler;req.targets={target};
 auto run=[&](){return run_prewarm({req},out.string()+"/","",7,"",false);};
 assert(run()==0 && attempts==1);assert(summary[0]==1 && summary[1]==0 && summary[2]==1 && summary[3]==0);
 assert(run()==0 && attempts==1);assert(summary[0]==0 && summary[2]==0 && summary[3]==1);
 // Real source and transitive dependency edits invalidate despite constant build_key=7.
 std::ofstream(src,std::ios::app)<<"// source revision\n";assert(run()==0 && attempts==2);
 std::ofstream(header)<<"#define VALUE 3\n";assert(run()==0 && attempts==3);
 req.targets[0].defines={"-DNEW_INPUT=1"};assert(run()==0 && attempts==4);
 // A new header earlier on the search path must invalidate even though old deps are unchanged.
 fs::create_directories(root/"earlier");req.targets[0].includes="-I"+(root/"earlier").string()+" "+target.includes;
 assert(run()==0 && attempts==5);assert(run()==0 && attempts==5);
 std::ofstream(root/"earlier/header.h")<<"#define VALUE 4\n";assert(run()==0 && attempts==6);
 std::ofstream(compiler,std::ios::app)<<"# toolchain revision\n";assert(run()==0 && attempts==7);
 setenv("TT_METAL_KERNEL_PREWARM_RETRY_FAILED","1",1);assert(run()==0 && attempts==8);
 unsetenv("TT_METAL_KERNEL_PREWARM_RETRY_FAILED");assert(run()==0 && attempts==8);
 // Compile args were stale: a compatible header fixes the real static assertion without a new key.
 std::ofstream(root/"earlier/header.h")<<"#define VALUE 2\n";assert(run()==1 && attempts==9);
 assert(summary[0]==1 && summary[1]==1 && summary[2]==0 && summary[3]==0);
 // Success/failure/duplicate/mismatch accounting in the real loop.
 jit_server::CompileRequest failed=req;failed.kernel_name="failed";failed.targets[0].defines.push_back("-DVALUE=0");
 fs::path badsrc=root/"bad.cpp";std::ofstream(badsrc)<<"static_assert(false, \"bad\");\n";failed.targets[0].srcs={badsrc};
 auto other=req;other.build_key=9;
 assert(run_prewarm({req,req,failed,other},out.string()+"/","",7,"",false)==1);
 assert(summary[0]==2 && summary[1]==1 && summary[2]==1 && summary[3]==0 && summary[5]==1 && summary[7]==1);
 // An unknown wrapper never results in an unsafe persistent skip.
 failed.gpp="/usr/bin/c++";assert(run_prewarm({failed},out.string()+"/","",7,"",false)==0);
 assert(summary[0]==1 && summary[2]==1 && summary[3]==0);
 std::cout<<"integration: stale arguments reproduced; unchanged skip, actual source/header/shadow/recipe/toolchain invalidation, explicit retry, success and exact summary counts passed\n";
}
"""

class PrewarmFailureTests(unittest.TestCase):
    def test_production_context_and_batch_loop(self):
        source = (ROOT / "tt_metal/impl/program/kernel_prewarm.cpp").read_text()
        context_and_loop = source[source.index("void hash_field(") : source.index("void initialize_prewarm_shutdown_dependencies()")]
        dependencies = (ROOT / "tt_metal/jit_build/depend.cpp").read_text()
        parser = dependencies[dependencies.index("ParsedDependencies parse_dependency_file(") : dependencies.index("\nnamespace {")]
        harness = INTEGRATION_PREFIX + "namespace tt::jit_build {" + parser + "}\n" + context_and_loop + INTEGRATION_MAIN
        with tempfile.TemporaryDirectory(prefix="prewarm-integration-") as work:
            path = Path(work)
            (path / "test.cpp").write_text(harness)
            subprocess.run(["c++", "-std=c++20", "-Wall", "-Wextra", "-Werror", "-pthread", "-I", str(ROOT), str(path / "test.cpp"), "-o", str(path / "test")], check=True, timeout=60)
            subprocess.run([str(path / "test"), str(path / "state")], check=True, timeout=60)

    def test_native_failure_policy(self):
        with tempfile.TemporaryDirectory(prefix="prewarm-policy-") as work:
            path = Path(work)
            source = path / "test.cpp"
            # <atomic> belongs to the test's concurrent publication scenario.
            source.write_text("#include <atomic>\n" + NATIVE)
            binary = path / "test"
            subprocess.run(
                ["c++", "-std=c++20", "-Wall", "-Wextra", "-Werror", "-pthread", "-I", str(ROOT), str(source), "-o", str(binary)],
                check=True,
                timeout=60,
            )
            subprocess.run([str(binary), str(path / "state")], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
