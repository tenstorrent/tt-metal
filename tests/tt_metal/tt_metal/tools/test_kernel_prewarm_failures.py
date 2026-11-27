# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only native regression of the production prewarm failure policy.

Compiles the real host-only policy header. No device, weights, or Metal library is loaded.
The integration harness additionally compiles run_prewarm directly from its production source,
with controlled build outcomes and lightweight executor/manifest dependencies.
"""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]

NATIVE = r"""
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
        {"/bin/sh", "-c", "echo 'x: error: static assertion failed'; echo 'fatal error: Killed signal'; exit 1"},
        {"/bin/sh", "-c", "echo 'x: error: static assertion failed'; echo 'No space left on device'; exit 1"},
        {"/bin/sh", "-c", "echo 'x: error: cannot open output file'; exit 1"},
        {"/bin/sh", "-c", "echo 'x: error: static assertion failed'; echo 'Read-only file system'; exit 1"}}) {
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
    middle.consume("x: error: static assertion failed\n" + std::string(20000, 'a'));
    middle.consume("No space left on device");
    middle.consume(std::string(20000, 'z'));
    middle.exit_code = 1;
    assert(!middle.definite_compile_failure());
    std::cout << "policy: retry, invalidation, reset, success, interruption, atomic state, concurrency passed; "
              << noisy.bytes << " output bytes retained in " << noisy.head.size() + noisy.tail.size() << " bytes\n";
}
"""


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
#include <utility>
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
 uint64_t build_key=7; std::string kernel_name,gpp,tree_commit;
 std::vector<jit_build::TargetRecipe> targets;
 std::vector<jit_build::GeneratedFile> generated_files;
}; }
std::vector<size_t> summary;
template<class T> size_t to_size(const T& v) { if constexpr (std::is_arithmetic_v<std::decay_t<T>>) return static_cast<size_t>(v); else return 0; }
template<class... Args> void log_warning(int,const char*,Args&&...) {}
template<class... Args> void log_error(int,const char*,Args&&...) {}
template<class... Args> void log_info(int,const char* text,Args&&... args) {
 if(std::string_view(text).starts_with("kernel prewarm: attempted=")) summary={to_size(args)...};
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
bool fast_current=false; std::atomic<int> current_checks{0};
bool target_is_current(const std::string& dir,const jit_build::TargetRecipe&){++current_checks;return fast_current && fs::exists(dir+"complete");}
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


REVIEW_MAIN = r"""
#include <sys/resource.h>
#include <sys/wait.h>
int main(int argc,char** argv) {
 assert(argc==4); fs::path root=argv[1];fs::create_directories(root);
 std::string mode=argv[2]; int expected=std::stoi(argv[3]);
 if(mode=="stale") {
  // A build_key captured by an older commit of this tree path: most recipes fail against the new
  // sources (cs04, ltx-rt 6796f8d2b3e: 21409 of 26200). The loop stops after the cutoff instead of
  // replaying all of them, also when some old kernels still build (mode 3); a key whose successes
  // outnumber half its failures is live (mode 1) and is never cut; mode 2 disables the cutoff.
  const bool live=expected==1, disabled=expected==2, mixed=expected==3;
  if(disabled)setenv("TT_METAL_KERNEL_PREWARM_STALE_LIMIT","0",1);
  std::vector<jit_server::CompileRequest> requests;
  const int oks=live ? 300 : mixed ? 50 : 0;
  for(int i=0;i<oks;++i) {
   jit_server::CompileRequest ok;ok.kernel_name="live"+std::to_string(i);ok.gpp="batch";
   jit_build::TargetRecipe target;target.target_name="first";ok.targets={target};requests.push_back(ok);
  }
  for(int i=0;i<400;++i) {
   jit_server::CompileRequest req;req.kernel_name="stale"+std::to_string(i);req.gpp="batch-stale";
   jit_build::TargetRecipe target;target.target_name="first";req.targets={target};requests.push_back(req);
  }
  auto count=run_prewarm(requests,root.string()+"/","",7,"",false);
  if(live || disabled) {
   assert(count==size_t(oks) && summary[2]==400 && summary[11]==0 && attempts==int(requests.size()));
  } else {
   assert(count==size_t(oks) && summary[2]==256 && summary[11]==144 && attempts==oks+256);
   assert(!fs::exists(root/"stale300"));
  }
  std::cout<<"stale build_key: mode="<<expected<<" failed="<<summary[2]<<" abandoned="<<summary[11]<<" passed\n";
  return 0;
 }
 if(mode=="stop") {
  // TT_METAL_KERNEL_PREWARM_STOP_FILE: a worker giving its device back creates the file, and the batch
  // takes no new kernel or target after it, so every join (compile barrier, exit) returns at once.
  // expected: 0 = no variable, 1 = variable set but the file never appears, 2 = the file appears while
  // kernel 4's first target builds (its second target and kernels 5-19 are skipped), 3 = it exists
  // before the batch starts.
  const fs::path stop_file=root/"stop";fs::remove(stop_file);
  if(expected!=0)setenv("TT_METAL_KERNEL_PREWARM_STOP_FILE",stop_file.c_str(),1);
  if(expected==2)stop_after=9;
  if(expected==3)std::ofstream(stop_file)<<"stop";
  std::vector<jit_server::CompileRequest> requests;
  for(int i=0;i<20;++i) {
   jit_server::CompileRequest req;req.kernel_name="kernel"+std::to_string(i);req.gpp="batch-stop";
   jit_build::TargetRecipe first;first.target_name="first";auto second=first;second.target_name="second";
   req.targets={first,second};requests.push_back(req);
  }
  auto count=run_prewarm(requests,root.string()+"/","",7,"",false);
  if(expected<=1)assert(count==40 && summary[12]==0 && attempts==40);
  if(expected==2) {
   assert(count==9 && summary[12]==31 && attempts==9);
   assert(fs::exists(root/"kernel4/first/complete") && !fs::exists(root/"kernel4/second/complete"));
   assert(!fs::exists(root/"kernel5/first/complete"));
  }
  if(expected==3)assert(count==0 && summary[12]==40 && attempts==0);
  for(int i=0;i<20;++i) {
   failure::KernelLock released(root/("kernel"+std::to_string(i))/".prewarm.lock");
   assert(released.acquired());
  }
  std::cout<<"stop file: mode="<<expected<<" built="<<count<<" stopped="<<summary[12]<<" passed\n";
  return 0;
 }
 if(mode=="batch") {
  struct rlimit limit{32,32};assert(setrlimit(RLIMIT_NOFILE,&limit)==0);
  fail_generated=expected==3;
  std::vector<jit_server::CompileRequest> requests;
  for(int i=0;i<96;++i) {
   jit_server::CompileRequest req;req.kernel_name="kernel"+std::to_string(i);req.gpp=expected==2 ? "batch-fail" : "batch";
   jit_build::TargetRecipe target;target.target_name="first";req.targets={target};
   requests.push_back(req);requests.push_back(req); // duplicate target
   req.targets[0].target_name="second";requests.push_back(req); // same kernel, distinct request
  }
  auto run=[&](){
   auto count=run_prewarm(requests,root.string()+"/","",7,"",false);
   assert(summary[2]==size_t(expected==2 ? 96 : 0));
   assert(summary[10]==size_t(expected==3 ? 96 : 0));
   assert(count+summary[2]+summary[9]==192);
   assert(summary[5]==96);
   if(expected==0)assert(count==192);
   if(expected==2)assert(count==96 && summary[9]==0);
   if(expected==3)assert(count==0 && summary[9]==192);
  };
  if(expected==1) {
   pid_t child=fork();assert(child>=0);
   run();
   if(child==0)_exit(0);
   int status;assert(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
  } else run();
  for(int i=0;i<96;++i)for(const char* target:{"first","second"}) {
   bool success=expected!=3 && !(expected==2 && std::string(target)=="second");
   assert(fs::exists(root/("kernel"+std::to_string(i))/target/"complete")==success);
   failure::KernelLock released(root/("kernel"+std::to_string(i))/".prewarm.lock");
   assert(released.acquired());
  }
  std::cout<<"96 kernels, 192 distinct targets, RLIMIT_NOFILE=32; concurrent="<<expected<<" passed\n";
  return 0;
 }
 fs::create_directories(root/"sfpi/compiler/bin");fs::path compiler=root/"sfpi/compiler/bin/g++";
 if(!fs::exists(compiler)) {
  std::ofstream(compiler)<<"#!/bin/sh\nexec /usr/bin/c++ \"$@\"\n";
  fs::permissions(compiler,fs::perms::owner_all);
  std::ofstream(root/"source.cpp")<<"static_assert(false, \"Index out of range\");\n";
 }
 jit_build::TargetRecipe target;target.target_name="target";target.compiler_opt_level="O0";
 target.srcs={(root/"source.cpp").string()};target.objs={"target.o"};
 jit_server::CompileRequest req;req.kernel_name="kernel";req.gpp=compiler;req.targets={target};
 assert(run_prewarm({req},(root/"cache").string()+"/","",7,"",false)==0);
 assert(attempts==expected);assert(summary[3]==size_t(1-expected));
 std::cout<<"cross-invocation attempts="<<attempts<<" skipped="<<summary[3]<<" passed\n";
}
"""


def review_prefix():
    """INTEGRATION_PREFIX with one active worker and a build_target that locks, fails or completes."""
    # One active worker makes dispatch-time descriptor retention deterministic.
    prefix = INTEGRATION_PREFIX.replace("std::launch::async", "std::launch::deferred").replace(
        "void write_generated_files(", "bool fail_generated=false;\nint stop_after=0;\nvoid write_generated_files("
    )
    prefix = prefix.replace(
        " ++attempts; fs::create_directories(dir);",
        r""" ++attempts; fs::create_directories(dir);
 if(gpp.starts_with("batch")) {
  failure::KernelLock competing(fs::path(dir).parent_path().parent_path()/".prewarm.lock");
  assert(!competing.acquired());
  int fd=::open((dir+"active").c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);assert(fd>=0);
  usleep(1000);assert(::close(fd)==0);assert(::unlink((dir+"active").c_str())==0);
  if((gpp=="batch-fail" && t.target_name=="second") || gpp=="batch-stale")throw std::runtime_error("injected build failure");
  if(gpp=="batch-stop" && attempts==stop_after)std::ofstream(getenv("TT_METAL_KERNEL_PREWARM_STOP_FILE"))<<"stop";
  std::ofstream(dir+"complete")<<"ok";return;
 }""",
    )
    return prefix.replace(
        "{fs::create_directories(dir);for(const auto& f:files)",
        '{failure::KernelLock competing(dir+".prewarm.lock");assert(!competing.acquired());if(fail_generated)throw std::runtime_error("injected genfiles failure");fs::create_directories(dir);for(const auto& f:files)',
    )


OFFLINE_STUBS = r"""
#include <algorithm>
#include <cctype>
#include <chrono>
#include <thread>
#include <sys/resource.h>
std::vector<jit_server::CompileRequest> g_manifest;
std::vector<jit_server::CompileRequest> read_manifest(const std::string&) { return g_manifest; }
std::once_flag g_launch_once;
std::mutex g_capture_mutex;
std::string g_tree_commit, g_capture_path;
std::unordered_set<std::string> g_captured_keys, g_prewarm_kernel_names;
std::atomic<bool> g_prewarm_names_ready{false}, g_batch_launched{false};
std::jthread g_prewarm_thread;
bool prewarm_globally_disabled() { return !env_truthy("TT_METAL_KERNEL_PREWARM"); }
void initialize_prewarm_shutdown_dependencies() {}
void wait_for_prewarm() {}
"""

OFFLINE_MAIN = r"""
const std::string C(40, 'c'), D(40, 'd'), TREE = "batch";  // gpp "batch" / "batch-stale" contain the tree path
fs::path root;
jit_server::CompileRequest req(uint64_t bk, const std::string& name, const std::string& commit, bool good = true) {
 jit_server::CompileRequest r; r.build_key = bk; r.kernel_name = name; r.gpp = good ? "batch" : "batch-stale";
 r.tree_commit = commit; jit_build::TargetRecipe t; t.target_name = "first"; r.targets = {t}; return r;
}
// key 7: the live key, captured by commit C, plus the same kernels as an older commit D left them (stale
// recipes), plus one recipe of another source tree. key 8: an old key without commits. key 9: stamped, all D.
std::vector<jit_server::CompileRequest> mixed_manifest() {
 std::vector<jit_server::CompileRequest> m;
 for (int i = 0; i < 3; ++i) m.push_back(req(7, "k" + std::to_string(i), C));
 for (int i = 0; i < 3; ++i) m.push_back(req(7, "k" + std::to_string(i), D, false));
 auto foreign = req(7, "f0", C); foreign.gpp = "/other/tree/g++"; m.push_back(foreign);
 for (int i = 10; i < 12; ++i) m.push_back(req(8, "k" + std::to_string(i), ""));
 for (int i = 20; i < 22; ++i) m.push_back(req(9, "k" + std::to_string(i), D, false));
 return m;
}
fs::path fresh(const std::string& name) { fs::path d = root / name; fs::remove_all(d); fs::create_directories(d); attempts = 0; return d; }
void stamp(const fs::path& d, uint64_t bk, int days_old) {
 fs::path p = d / ("tt-metal-cache" + std::to_string(bk)); fs::create_directories(p);
 std::ofstream(p / "kernel_prewarm.live") << "";
 fs::last_write_time(p / "kernel_prewarm.live", fs::file_time_type::clock::now() - std::chrono::hours(24 * days_old));
}
size_t offline(const fs::path& d) { return prewarm_manifest_offline((d / "tt-metal-cache").string(), TREE); }
bool built(const fs::path& d, uint64_t bk, const std::string& k) {
 return fs::exists(d / ("tt-metal-cache" + std::to_string(bk)) / "kernels" / k / "first" / "complete");
}
void clear_env() {
 for (const char* k : {"TT_METAL_KERNEL_PREWARM_LIVE_DAYS", "TT_METAL_KERNEL_PREWARM_ALL_KEYS",
                       "TT_METAL_KERNEL_PREWARM_STALE_LIMIT", "TT_METAL_KERNEL_PREWARM_TREE_COMMIT"}) unsetenv(k);
}

void filters() {
 g_manifest = mixed_manifest();
 {  // live stamps on keys 7 and 9 and the current commit C: only key 7's own recipes build
  clear_env(); setenv("TT_METAL_KERNEL_PREWARM_TREE_COMMIT", C.c_str(), 1);
  auto d = fresh("live"); stamp(d, 7, 0); stamp(d, 9, 0);
  assert(offline(d) == 3 && attempts == 3);
  for (const char* k : {"k0", "k1", "k2"}) assert(built(d, 7, k));
  assert(!built(d, 7, "f0") && !built(d, 8, "k10") && !built(d, 9, "k20"));
 }
 {  // ALL_KEYS builds the unstamped legacy key too; entries of another commit stay skipped
  setenv("TT_METAL_KERNEL_PREWARM_ALL_KEYS", "1", 1);
  auto d = fresh("allkeys"); stamp(d, 7, 0); stamp(d, 9, 0);
  assert(offline(d) == 5 && attempts == 5 && built(d, 8, "k10") && built(d, 8, "k11") && !built(d, 9, "k20"));
 }
 {  // no stamp anywhere (cache from before the stamps): every key builds, still only its own commit
  clear_env(); setenv("TT_METAL_KERNEL_PREWARM_TREE_COMMIT", C.c_str(), 1);
  auto d = fresh("nostamp"); assert(offline(d) == 5 && attempts == 5);
 }
 {  // a stamp older than LIVE_DAYS (default 14) counts as no stamp
  auto d = fresh("old"); stamp(d, 7, 0); stamp(d, 8, 20);
  assert(offline(d) == 3 && attempts == 3 && !built(d, 8, "k10"));
  setenv("TT_METAL_KERNEL_PREWARM_LIVE_DAYS", "30", 1);
  d = fresh("old30"); stamp(d, 7, 0); stamp(d, 8, 20);
  assert(offline(d) == 5 && attempts == 5 && built(d, 8, "k10"));
 }
 {  // current commit unknown: the commit filter is inert, the stale recipes of key 9 are attempted and fail
  clear_env();
  auto d = fresh("unknown"); stamp(d, 7, 0); stamp(d, 9, 0);
  assert(offline(d) == 3 && attempts == 5 && !built(d, 9, "k20"));
 }
 {  // an entry without a commit never counts as another commit's, whatever the current one
  g_manifest = {req(8, "k10", ""), req(8, "k11", "")};
  setenv("TT_METAL_KERNEL_PREWARM_TREE_COMMIT", D.c_str(), 1);
  auto d = fresh("legacy"); assert(offline(d) == 2 && attempts == 2);
 }
}

void stale_limit() {
 // Old-format manifest (no commits): 1 good + 400 stale recipes. The stale limit is the only cut.
 auto stale_manifest = [&]() {
  std::vector<jit_server::CompileRequest> m{req(7, "good", "")};
  for (int i = 0; i < 400; ++i) m.push_back(req(7, "s" + std::to_string(i), "", false));
  return m;
 };
 clear_env(); g_manifest = stale_manifest();
 auto d = fresh("stale"); assert(offline(d) == 1 && attempts == 257);
 setenv("TT_METAL_KERNEL_PREWARM_STALE_LIMIT", "0", 1);
 d = fresh("nolimit"); assert(offline(d) == 1 && attempts == 401);
 // The same manifest with commits: the current commit's recipes build and the other commit's are never tried.
 clear_env(); setenv("TT_METAL_KERNEL_PREWARM_TREE_COMMIT", C.c_str(), 1);
 g_manifest = {req(7, "good", C)};
 for (int i = 0; i < 400; ++i) g_manifest.push_back(req(7, "s" + std::to_string(i), D, false));
 d = fresh("stale-commit"); assert(offline(d) == 1 && attempts == 1);
}

void write_file(const fs::path& p, const std::string& text) { fs::create_directories(p.parent_path()); std::ofstream(p) << text; }

// An already-built target with no recorded failure takes the fast path (no failure fingerprint, so no
// g++ -M include scan); one with a recorded failure still goes through the failure policy.
void current_fast_path() {
 clear_env(); setenv("TT_METAL_KERNEL_PREWARM_TREE_COMMIT", C.c_str(), 1);
 g_manifest = {req(7, "k0", C), req(7, "k1", C), req(7, "k2", C)};
 auto d = fresh("current");
 assert(offline(d) == 3 && attempts == 3 && current_checks == 3);  // cold: checked, not current, built
 fast_current = true; attempts = 0; current_checks = 0;
 assert(offline(d) == 3 && current_checks == 3 && attempts == 3 && built(d, 7, "k1"));  // warm: fast path
 fs::path kdir = d / "tt-metal-cache7" / "kernels" / "k1";
 write_file(kdir.string() + "/.prewarm-failed-first", "prewarm-failure-v1 other\n");
 current_checks = 0; assert(offline(d) == 3 && current_checks == 2);  // k1's recorded failure: regular path
 // a fast-path build that throws falls through to the failure policy, which builds again (attempts == 2) and counts it failed
 g_manifest = {req(7, "k9", C, false)};
 auto e = fresh("current-throw"); fast_current = true;
 fs::create_directories(e / "tt-metal-cache7" / "kernels" / "k9" / "first"); write_file((e / "tt-metal-cache7" / "kernels" / "k9" / "first" / "complete").string(), "ok");
 current_checks = 0; assert(offline(e) == 0 && current_checks == 1 && attempts == 2);
}


void commit_sources() {
 const std::string A(40, 'a'), B(40, 'b'), E(40, 'e'), SHA256(64, '1');
 clear_env();
 auto t = fresh("commit");
 assert(current_tree_commit((t / "none").string() + "/").empty() && current_tree_commit("").empty());
 write_file(t / "detached/.git/HEAD", A + "\n"); assert(current_tree_commit((t / "detached").string() + "/") == A);
 write_file(t / "loose/.git/HEAD", "ref: refs/heads/main\n"); write_file(t / "loose/.git/refs/heads/main", B + "\n");
 assert(current_tree_commit((t / "loose").string() + "/") == B);
 write_file(t / "packed/.git/HEAD", "ref: refs/heads/ltx-rt\n");
 write_file(t / "packed/.git/packed-refs", "# pack-refs\n" + std::string(40, '1') + " refs/heads/other\n" + E + " refs/heads/ltx-rt\n");
 assert(current_tree_commit((t / "packed").string() + "/") == E);
 write_file(t / "sha256/.git/HEAD", SHA256 + "\n"); assert(current_tree_commit((t / "sha256").string() + "/") == SHA256);
 // A linked worktree: .git is a file and the branch lives in the common dir.
 write_file(t / "wt/.git", "gitdir: " + (t / "main.git/worktrees/wt").string() + "\n");
 write_file(t / "main.git/worktrees/wt/HEAD", "ref: refs/heads/wt\n"); write_file(t / "main.git/worktrees/wt/commondir", "../..\n");
 write_file(t / "main.git/refs/heads/wt", A + "\n"); assert(current_tree_commit((t / "wt").string() + "/") == A);
 // A deployed slot without .git: the stamp written after each deploy, first word.
 write_file(t / "slot/.ltx-build-stamp", B + " 2026-11-16 deploy\n"); assert(current_tree_commit((t / "slot").string() + "/") == B);
 write_file(t / "junk/.ltx-build-stamp", "not-a-commit\n"); assert(current_tree_commit((t / "junk").string() + "/").empty());
 write_file(t / "unborn/.git/HEAD", "ref: refs/heads/none\n"); assert(current_tree_commit((t / "unborn").string() + "/").empty());
 // The override wins, and an invalid one means unknown rather than falling back to another source.
 setenv("TT_METAL_KERNEL_PREWARM_TREE_COMMIT", E.c_str(), 1); assert(current_tree_commit((t / "detached").string() + "/") == E);
 setenv("TT_METAL_KERNEL_PREWARM_TREE_COMMIT", "xyz", 1); assert(current_tree_commit((t / "detached").string() + "/").empty());
 { std::string up = E; for (auto& ch : up) ch = std::toupper(ch); setenv("TT_METAL_KERNEL_PREWARM_TREE_COMMIT", up.c_str(), 1); assert(current_tree_commit((t / "detached").string() + "/") == E); }
 assert(from_other_commit(A, B) && !from_other_commit(A, A) && !from_other_commit("", B) && !from_other_commit(A, ""));
}

// The device-open path itself (call_once, so one scenario per process): it writes the live stamp, records
// the commit on captured entries and seeds the capture set from this commit's entries only.
void open_path(bool cold) {
 clear_env(); setenv("TT_METAL_KERNEL_PREWARM", "1", 1); setenv("TT_METAL_KERNEL_PREWARM_TREE_COMMIT", C.c_str(), 1);
 auto d = fresh(cold ? "open-cold" : "open");
 const std::string kernels = (d / "tt-metal-cache7" / "kernels").string() + "/";
 const fs::path live = d / "tt-metal-cache7" / "kernel_prewarm.live";
 assert(!fs::exists(live));
 if (!cold) {
  write_file(d / "kernel_prewarm.manifest", "x");  // non-empty: read_manifest is the stub below
  g_manifest = {req(9, "a0", C), req(9, "a1", C), req(9, "b0", D), req(9, "b1", D), req(9, "c0", "")};
 }
 maybe_launch_prewarm(kernels, "", 7, TREE);
 assert(fs::exists(live));
 assert(fs::file_time_type::clock::now() - fs::last_write_time(live) < std::chrono::seconds(60));
 assert(g_tree_commit == C && g_capture_path == (d / "kernel_prewarm.manifest").string());
 if (!cold) {
  assert(g_captured_keys.count("9/a0") && g_captured_keys.count("9/a1") && g_captured_keys.count("9/c0"));
  assert(!g_captured_keys.count("9/b0") && !g_captured_keys.count("9/b1"));
  assert(!g_batch_launched);  // the manifest has nothing of build_key 7
 }
 // The offline pass reads exactly that stamp: key 7 (stamped by the open) builds, the unstamped key 8 does not.
 g_manifest = mixed_manifest();
 assert(offline(d) == 3 && attempts == 3 && built(d, 7, "k0") && !built(d, 8, "k10"));
}

int main(int, char** argv) {
 root = argv[1]; fs::create_directories(root); std::string mode = argv[2];
 if (mode == "filters") filters();
 else if (mode == "stale") stale_limit();
 else if (mode == "current") current_fast_path();
 else if (mode == "commit") commit_sources();
 else open_path(mode == "open-cold");
 std::cout << "offline filters: " << mode << " passed\n";
}
"""


class PrewarmFailureTests(unittest.TestCase):
    def test_production_context_and_batch_loop(self):
        source = (ROOT / "tt_metal/impl/program/kernel_prewarm.cpp").read_text()
        context_and_loop = source[
            source.index("void hash_field(") : source.index("void initialize_prewarm_shutdown_dependencies()")
        ]
        dependencies = (ROOT / "tt_metal/jit_build/depend.cpp").read_text()
        parser = dependencies[
            dependencies.index("ParsedDependencies parse_dependency_file(") : dependencies.index("\nnamespace {")
        ]
        harness = (
            INTEGRATION_PREFIX + "namespace tt::jit_build {" + parser + "}\n" + context_and_loop + INTEGRATION_MAIN
        )
        with tempfile.TemporaryDirectory(prefix="prewarm-integration-") as work:
            path = Path(work)
            (path / "test.cpp").write_text(harness)
            subprocess.run(
                [
                    "c++",
                    "-std=c++20",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pthread",
                    "-I",
                    str(ROOT),
                    str(path / "test.cpp"),
                    "-o",
                    str(path / "test"),
                ],
                check=True,
                timeout=60,
            )
            subprocess.run([str(path / "test"), str(path / "state")], check=True, timeout=60)

    def test_review_regressions(self):
        source = (ROOT / "tt_metal/impl/program/kernel_prewarm.cpp").read_text()
        production = source[
            source.index("void hash_field(") : source.index("void initialize_prewarm_shutdown_dependencies()")
        ]
        dependencies = (ROOT / "tt_metal/jit_build/depend.cpp").read_text()
        parser = dependencies[
            dependencies.index("ParsedDependencies parse_dependency_file(") : dependencies.index("\nnamespace {")
        ]
        prefix = review_prefix()
        with tempfile.TemporaryDirectory(prefix="prewarm-review-") as work:
            path = Path(work)
            (path / "test.cpp").write_text(
                prefix + "namespace tt::jit_build {" + parser + "}\n" + production + REVIEW_MAIN
            )
            subprocess.run(
                [
                    "c++",
                    "-std=c++20",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pthread",
                    "-I",
                    str(ROOT),
                    str(path / "test.cpp"),
                    "-o",
                    str(path / "test"),
                ],
                check=True,
                timeout=60,
            )
            for concurrent in (0, 1, 2, 3):
                with self.subTest(batch_concurrent=concurrent):
                    subprocess.run(
                        [str(path / "test"), str(path / f"batch{concurrent}"), "batch", str(concurrent)],
                        check=True,
                        timeout=30,
                    )
            for case in (0, 1, 2, 3):
                with self.subTest(stop_file=case):
                    env = os.environ.copy()
                    env.pop("TT_METAL_KERNEL_PREWARM_STOP_FILE", None)
                    subprocess.run(
                        [str(path / "test"), str(path / f"stop{case}"), "stop", str(case)],
                        env=env,
                        check=True,
                        timeout=30,
                    )
            for case in (0, 1, 2, 3):
                with self.subTest(stale_build_key=case):
                    env = os.environ.copy()
                    env.pop("TT_METAL_KERNEL_PREWARM_STALE_LIMIT", None)
                    subprocess.run(
                        [str(path / "test"), str(path / f"stale{case}"), "stale", str(case)],
                        env=env,
                        check=True,
                        timeout=30,
                    )
            env = os.environ.copy()
            for key in (
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "GCC_EXEC_PREFIX",
                "COMPILER_PATH",
                "TT_METAL_KERNEL_PREWARM_RETRY_FAILED",
            ):
                env.pop(key, None)
            env["LANG"] = "C"
            command = [str(path / "test"), str(path / "environment"), "environment"]
            subprocess.run(command + ["1"], env=env, check=True, timeout=30)
            for key in (
                "INVOCATION_ID",
                "SYSTEMD_EXEC_PID",
                "SSH_CONNECTION",
                "SSH_CLIENT",
                "SSH_TTY",
                "JOURNAL_STREAM",
                "XDG_SESSION_ID",
            ):
                env[key] = "changed-session-metadata"
                with self.subTest(unrelated_environment=key):
                    subprocess.run(command + ["0"], env=env, check=True, timeout=30)
            for key, value in (("CPATH", str(path)), ("LANG", "C.UTF-8"), ("UNKNOWN_COMPILER_INPUT", "changed")):
                env[key] = value
                with self.subTest(relevant_environment=key):
                    subprocess.run(command + ["1"], env=env, check=True, timeout=30)
                    subprocess.run(command + ["0"], env=env, check=True, timeout=30)

    def test_offline_filters_commit_and_live_stamp(self):
        """Offline live-stamp filter, the per-entry commit filter and the real device-open path, host-only."""
        source = (ROOT / "tt_metal/impl/program/kernel_prewarm.cpp").read_text()
        stamp_helpers = source[
            source.index("std::string default_manifest_path(") : source.index("// Build one target from a portable recipe")
        ]
        production = source[
            source.index("void hash_field(") : source.index("void initialize_prewarm_shutdown_dependencies()")
        ]
        open_path = source[source.index("void maybe_launch_prewarm(") : source.index("std::size_t prewarm_manifest_offline(")]
        offline = source[source.index("std::size_t prewarm_manifest_offline(") : source.index("void wait_for_prewarm()")]
        dependencies = (ROOT / "tt_metal/jit_build/depend.cpp").read_text()
        parser = dependencies[
            dependencies.index("ParsedDependencies parse_dependency_file(") : dependencies.index("\nnamespace {")
        ]
        harness = (
            review_prefix()
            + "namespace tt::jit_build {"
            + parser
            + "}\n"
            + OFFLINE_STUBS.split("std::vector<jit_server::CompileRequest> g_manifest;")[0]
            + stamp_helpers
            + production
            + "std::vector<jit_server::CompileRequest> g_manifest;"
            + OFFLINE_STUBS.split("std::vector<jit_server::CompileRequest> g_manifest;")[1]
            + open_path
            + offline
            + OFFLINE_MAIN
        )
        with tempfile.TemporaryDirectory(prefix="prewarm-offline-") as work:
            path = Path(work)
            (path / "test.cpp").write_text(harness)
            subprocess.run(
                [
                    "c++",
                    "-std=c++20",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pthread",
                    "-I",
                    str(ROOT),
                    str(path / "test.cpp"),
                    "-o",
                    str(path / "test"),
                ],
                check=True,
                timeout=120,
            )
            env = os.environ.copy()
            for key in [k for k in env if k.startswith("TT_METAL_KERNEL_")]:
                env.pop(key)
            for mode in ("filters", "stale", "commit", "current", "open", "open-cold"):
                with self.subTest(mode=mode):
                    subprocess.run([str(path / "test"), str(path / mode), mode], env=env, check=True, timeout=120)

    def test_manifest_round_trips_tree_commit(self):
        """The real append_manifest_entry and read_manifest carry treeCommit (needs a capnp build tree)."""
        build = Path(os.environ.get("TT_METAL_CAPNP_BUILD_DIR", "/home/smarton/tt-metal-ltx-rt/build_Release"))
        capnp_dir = build / "_deps/capnproto-build/c++/src"
        needed = [build / "bin/capnp", build / "bin/capnpc-c++", capnp_dir / "capnp/libcapnp.a", capnp_dir / "capnp/libcapnp-rpc.a", capnp_dir / "kj/libkj.a", capnp_dir / "kj/libkj-async.a"]
        if not all(f.exists() for f in needed):
            self.skipTest("no built capnp (set TT_METAL_CAPNP_BUILD_DIR to a tt-metal build tree)")
        source = (ROOT / "tt_metal/impl/program/kernel_prewarm.cpp").read_text()
        manifest_state = source[source.index("std::mutex g_manifest_mutex;") : source.index("// Prewarm orchestration.")]
        append = source[source.index("void append_manifest_entry(") : source.index("void maybe_launch_prewarm(")]
        harness = r"""
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <cassert>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>
#include "impl/jit_server/rpc.capnp.h"
#include "impl/jit_server/types.hpp"
namespace tt { struct LogT {}; inline LogT LogMetal; template <class... A> void log_warning(LogT, const char*, A&&...) {} }
namespace tt::tt_metal::kernel_prewarm {
namespace {
"""
        harness += manifest_state + "}\n"
        harness += (
            'const char* manifest_write_path() { return g_capture_path.empty() ? nullptr : g_capture_path.c_str(); }\n'
            + append
            + r"""
void run(const char* path) {
 const std::string C(40, 'c'), D(40, 'd');
 g_capture_path = path;
 for (const std::string& commit : {C, D, std::string()}) {
  g_tree_commit = commit;
  jit_server::CompileRequest r; r.build_key = 7; r.kernel_name = "k-" + commit.substr(0, 1) + "x"; r.gpp = "g++";
  jit_build::TargetRecipe t; t.target_name = "first"; r.targets = {t};
  append_manifest_entry(r);
 }
 auto m = read_manifest(path);
 assert(m.size() == 3);
 assert(m[0].tree_commit == C && m[1].tree_commit == D && m[2].tree_commit.empty());
 assert(m[0].build_key == 7 && m[0].kernel_name == "k-cx" && m[0].targets.size() == 1);
}
}
int main(int, char** argv) { tt::tt_metal::kernel_prewarm::run(argv[1]); return 0; }
"""
        )
        with tempfile.TemporaryDirectory(prefix="prewarm-roundtrip-") as work:
            path = Path(work)
            schema = path / "gen/impl/jit_server"
            schema.mkdir(parents=True)
            (schema / "rpc.capnp").write_text((ROOT / "tt_metal/impl/jit_server/rpc.capnp").read_text())
            subprocess.run(
                [str(build / "bin/capnp"), "compile", f"-I{build}/include", f"-o{build}/bin/capnpc-c++:{path}/gen",
                 "--src-prefix", str(path / "gen"), str(schema / "rpc.capnp")],
                check=True, timeout=60, cwd=path / "gen",
            )
            (path / "test.cpp").write_text(harness)
            subprocess.run(
                ["c++", "-std=c++20", "-Wall", "-Wextra", "-Werror", "-pthread", "-I", str(path / "gen"),
                 "-I", str(ROOT / "tt_metal"), "-I", str(build / "include"), str(path / "test.cpp"),
                 str(schema / "rpc.capnp.c++"), str(capnp_dir / "capnp/libcapnp-rpc.a"), str(capnp_dir / "capnp/libcapnp.a"),
                 str(capnp_dir / "kj/libkj-async.a"), str(capnp_dir / "kj/libkj.a"),
                 "-o", str(path / "test")],
                check=True, timeout=300,
            )
            subprocess.run([str(path / "test"), str(path / "kernel_prewarm.manifest")], check=True, timeout=30)

    def test_native_failure_policy(self):
        with tempfile.TemporaryDirectory(prefix="prewarm-policy-") as work:
            path = Path(work)
            source = path / "test.cpp"
            # <atomic> belongs to the test's concurrent publication scenario.
            source.write_text("#include <atomic>\n" + NATIVE)
            binary = path / "test"
            subprocess.run(
                [
                    "c++",
                    "-std=c++20",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pthread",
                    "-I",
                    str(ROOT),
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
                timeout=60,
            )
            subprocess.run([str(binary), str(path / "state")], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
