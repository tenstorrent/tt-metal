#!/usr/bin/env python3
"""Tests for the redundant-drain checker."""
import contextlib
import io
import os
import runpy
import sys
import textwrap
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "check_redundant_drain.py")


def run(*args):
    """Run the checker as a script (`__main__`, as the hook does), in-process."""
    argv = [SCRIPT, *map(str, args)]
    out, err, code = io.StringIO(), io.StringIO(), 0
    saved = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            runpy.run_path(SCRIPT, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv = saved
    return types.SimpleNamespace(
        returncode=code, stdout=out.getvalue(), stderr=err.getvalue()
    )


def hdr(tmp_path, name, body, arch="tt_llk_blackhole"):
    """Write a header inside a fake architecture tree so the tree lookup works."""
    d = tmp_path / arch / "llk_lib"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(textwrap.dedent(body))
    return p


def test_mop_sync_before_tensix_sync_is_redundant(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void start()
        {
            mop_sync();
            tensix_sync();
            mailbox_write(MathThreadId, READY);
        }
        """,
    )
    r = run(f)
    assert r.returncode == 0
    assert "a.h:4: REDUNDANT: mop_sync()" in r.stdout
    assert (
        r.stdout.count("a.h:") == 1
    )  # the tensix_sync that follows is required and not reported


def test_drain_before_mailbox_is_required(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void release()
        {
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            tensix_sync();
            mailbox_write(MathThreadId, DONE);
        }
        """,
    )
    assert run(f).stdout == ""


def test_drain_followed_only_by_tensix_pushes_is_advisory(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void reconfig()
        {
            tensix_sync();
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
            TT_SETDMAREG(0, 0, 0, LO_16(p_gpr_pack::TMP0));
        }
        """,
    )
    r = run(f)
    assert "a.h:4: ADVISORY: tensix_sync() nothing RISC-side follows" in r.stdout
    assert r.returncode == 0


def test_two_tensix_syncs_back_to_back_flags_the_first(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void twice()
        {
            tensix_sync();
            TTI_NOP;
            tensix_sync();
            semaphore_read(semaphore::MATH_PACK);
        }
        """,
    )
    r = run(f)
    assert "a.h:4: REDUNDANT: tensix_sync()" in r.stdout
    assert "a.h:6:" not in r.stdout


def test_tensix_sync_before_mop_sync_is_not_covered(tmp_path):
    # mop_sync is the weaker drain, so it does not subsume an earlier tensix_sync; the scan
    # continues past it and the mailbox write makes the first drain required.
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void mixed()
        {
            tensix_sync();
            TTI_NOP;
            mop_sync();
            mailbox_write(MathThreadId, DONE);
        }
        """,
    )
    assert "REDUNDANT" not in run(f).stdout


def test_mmio_write_through_pointer_is_a_consumer(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void program()
        {
            volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);
            mop_sync();
            mop_cfg[0] = m_outer_loop_len;
        }
        """,
    )
    assert run(f).stdout == ""


def test_helper_in_the_tree_is_classified_by_its_body(tmp_path):
    hdr(
        tmp_path,
        "helpers.h",
        """
        inline void writes_cfg_by_mmio()
        {
            volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();
            cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32] = 0;
        }
        inline void pushes_only()
        {
            TTI_SETADCXY(0b100, 0, 0, 0, 0, 0b1011);
            TTI_SETADCZW(0b100, 0, 0, 0, 0, 0b1111);
        }
        """,
    )
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void required()
        {
            tensix_sync();
            writes_cfg_by_mmio();
        }
        inline void advisory()
        {
            tensix_sync();
            pushes_only();
        }
        """,
    )
    r = run(f)
    assert "a.h:4:" not in r.stdout
    assert "a.h:9: ADVISORY" in r.stdout


def test_unknown_helper_suppresses_the_finding(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void maybe()
        {
            tensix_sync();
            something_not_in_the_tree();
        }
        """,
    )
    assert run(f).stdout == ""


def test_pointer_store_and_volatile_load_are_consumers(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void flag(std::uint32_t *done)
        {
            tensix_sync();
            *done = 1;
        }
        inline std::uint32_t poll(volatile std::uint32_t *p)
        {
            tensix_sync();
            return p[0];
        }
        """,
    )
    assert run(f).stdout == ""


def test_drain_inside_a_condition_is_not_a_statement(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void guarded(bool x)
        {
            if (x)
            {
                tensix_sync();
            }
            mailbox_write(MathThreadId, DONE);
        }
        """,
    )
    assert run(f).stdout == ""


def test_definitions_and_comments_are_ignored(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void tensix_sync()
        {
            store_blocking(&pc_buf_base[1], 0);
        }
        inline void documented()
        {
            // tensix_sync(); is not needed here
            TTI_NOP;
        }
        """,
    )
    assert run(f).stdout == ""


def test_quasar_is_skipped(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void q()
        {
            mop_sync();
            tensix_sync();
        }
        """,
        arch="tt_llk_quasar",
    )
    r = run(f)
    assert "REDUNDANT" not in r.stdout
    assert "Quasar" in r.stdout


def test_baseline_suppresses_a_reviewed_site(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void kept()
        {
            tensix_sync();
            TTI_NOP;
        }
        """,
    )
    rel = os.path.relpath(f).replace(os.sep, "/")
    b = tmp_path / "baseline.txt"
    b.write_text(
        f"{rel}:inline void kept()  # the caller's Tensix ops must see the RISC stores above\n"
    )
    assert run(f, "--baseline", b).stdout == ""
    assert "ADVISORY" in run(f).stdout


def test_json_output_and_strict_exit(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void start()
        {
            mop_sync();
            tensix_sync();
            mailbox_write(MathThreadId, READY);
        }
        """,
    )
    r = run(f, "--json")
    assert '"kind": "REDUNDANT"' in r.stdout and '"drain": "mop_sync"' in r.stdout
    assert run(f, "--strict").returncode == 1
    assert run(f).returncode == 0


def test_real_tree_runs_clean_of_errors():
    """The checker parses every real header; what it finds there is advisory, not asserted."""
    llk = os.path.normpath(os.path.join(HERE, "..", ".."))
    r = run(
        "--tree",
        os.path.join(llk, "tt_llk_blackhole"),
        "--tree",
        os.path.join(llk, "tt_llk_wormhole_b0"),
    )
    assert r.returncode == 0
    assert "Traceback" not in r.stderr


# ---------------------------------------------------------------------------------------------
# The repository outside the LLK trees: compute kernels, the compute API, thread-scoped code.


def repo(tmp_path):
    """A fake repository: both LLK trees, the compute API directory and a kernel directory."""
    r = tmp_path / "repo"
    for a in ("tt_llk_blackhole", "tt_llk_wormhole_b0"):
        (r / "tt_metal" / "tt-llk" / a / "llk_lib").mkdir(parents=True)
    (r / "tt_metal" / "hw" / "inc" / "api" / "compute").mkdir(parents=True)
    (r / "ttnn" / "op" / "kernels" / "compute").mkdir(parents=True)
    return r


def write(r, rel, body):
    p = r / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body))
    return p


def llk_helpers(r, body, name="llk_helpers.h"):
    """The same helper bodies in both architecture trees (a kernel is compiled for both)."""
    for a in ("tt_llk_blackhole", "tt_llk_wormhole_b0"):
        write(r, f"tt_metal/tt-llk/{a}/llk_lib/{name}", body)


KERNEL_HELPERS = """
    inline void llk_pop_tiles(std::uint32_t cb, std::uint32_t n)
    {
        cb_interface[cb].tiles_acked += n;
    }
    inline void llk_pack_store(std::uint32_t *p)
    {
        *p = 1;
    }
    inline void llk_math_push()
    {
        TTI_NOP;
    }
    """


def test_kernel_drain_on_one_thread_sees_only_that_thread(tmp_path):
    r = repo(tmp_path)
    llk_helpers(r, KERNEL_HELPERS)
    k = write(
        r,
        "ttnn/op/kernels/compute/k.cpp",
        """
        namespace NAMESPACE {
        void MAIN {
            UNPACK(tensix_sync());
            PACK((llk_pack_store(out)));
            MATH((llk_math_push()));
        }
        }
        """,
    )
    out = run("--repo", r, k).stdout
    assert (
        "k.cpp:4: ADVISORY" in out
    )  # the pack-thread store is not on the unpack thread
    k2 = write(
        r,
        "ttnn/op/kernels/compute/k2.cpp",
        """
        namespace NAMESPACE {
        void MAIN {
            UNPACK(tensix_sync());
            UNPACK((llk_pop_tiles(0, 1)));
        }
        }
        """,
    )
    assert run("--repo", r, k2).stdout == ""


def test_compile_for_trisc_drain_runs_on_every_thread(tmp_path):
    r = repo(tmp_path)
    llk_helpers(r, KERNEL_HELPERS)
    h = write(
        r,
        "ttnn/op/kernels/compute/sync.hpp",
        """
        FORCE_INLINE void sync_enter(volatile std::uint32_t *sem)
        {
        #if defined(COMPILE_FOR_TRISC)
            tensix_sync();
        #endif
            __atomic_fetch_add(&sem[0], 1, __ATOMIC_RELEASE);
        }
        FORCE_INLINE void sync_bare()
        {
        #if defined(COMPILE_FOR_TRISC)
            tensix_sync();
        #endif
            TTI_NOP;
        }
        """,
    )
    out = run("--repo", r, h).stdout
    assert "sync.hpp:5:" not in out
    assert "sync.hpp:12: ADVISORY" in out


def test_thread_regions_by_directive(tmp_path):
    r = repo(tmp_path)
    llk_helpers(r, KERNEL_HELPERS)
    h = write(
        r,
        "ttnn/op/kernels/compute/regions.hpp",
        """
        inline void a(std::uint32_t *p)
        {
        #ifdef UCK_CHLKC_UNPACK
            tensix_sync();
        #endif
        #ifdef UCK_CHLKC_PACK
            *p = 1;
        #endif
            TTI_NOP;
        }
        inline void b(std::uint32_t *p)
        {
        #ifdef UCK_CHLKC_UNPACK
            tensix_sync();
        #endif
        #if defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK)
            *p = 1;
        #endif
        }
        inline void c(std::uint32_t *p)
        {
        #if defined(LLK_TRISC_MATH)
            tensix_sync();
        #endif
        #if !defined(LLK_TRISC_PACK)
            *p = 1;
        #endif
        }
        """,
    )
    out = run("--repo", r, h).stdout
    assert "regions.hpp:5: ADVISORY" in out
    assert "regions.hpp:14:" not in out
    assert "regions.hpp:23:" not in out


def test_undecidable_condition_enumerates_both_paths(tmp_path):
    r = repo(tmp_path)
    h = write(
        r,
        "ttnn/op/kernels/compute/paths.hpp",
        """
        inline void f(std::uint32_t *p)
        {
            tensix_sync();
        #if FOO
            tensix_sync();
        #else
            *p = 1;
        #endif
        }
        inline void g(std::uint32_t *p)
        {
            tensix_sync();
        #if defined(FOO) && BAR
            TTI_NOP;
        #elif !defined(FOO)
            TTI_NOP;
        #else
            TTI_NOP;
        #endif
        }
        """,
    )
    out = run("--repo", r, h).stdout
    assert "paths.hpp:4:" not in out  # on the !FOO path the store needs the drain
    assert "paths.hpp:6: ADVISORY" in out  # on its own path nothing follows
    assert "paths.hpp:13: ADVISORY" in out  # every path pushes only


def test_too_many_undecidable_conditions_is_unknown(tmp_path):
    r = repo(tmp_path)
    body = ["inline void f()", "{", "    tensix_sync();"]
    for i in range(5):
        body += [f"#if A{i}", "    TTI_NOP;", "#endif"]
    body += ["}"]
    h = write(r, "ttnn/op/kernels/compute/many.hpp", "\n".join(body) + "\n")
    assert run("--repo", r, h).stdout == ""


def test_loop_replays_the_body_before_the_drain(tmp_path):
    r = repo(tmp_path)
    h = write(
        r,
        "ttnn/op/kernels/compute/loops.hpp",
        """
        inline void f(volatile std::uint32_t *p)
        {
            for (int i = 0; i < 4; i++)
            {
                p[0] = i;
                tensix_sync();
            }
        }
        inline void g()
        {
            for (int i = 0; i < 4; i++)
            {
                TTI_NOP;
                tensix_sync();
            }
        }
        inline void h(std::uint32_t *p)
        {
            for (int i = 0; i < 4; i++)
            {
                tensix_sync();
                TTI_NOP;
            }
            *p = 1;
        }
        inline void k()
        {
            while (true)
            {
                tensix_sync();
                TTI_NOP;
                tensix_sync();
            }
        }
        """,
    )
    out = run("--repo", r, h).stdout
    assert "loops.hpp:7:" not in out  # the next iteration stores through p
    assert "loops.hpp:15: ADVISORY" in out
    assert "loops.hpp:22:" not in out  # the last iteration is followed by the store
    assert "loops.hpp:31: REDUNDANT" in out
    assert "loops.hpp:33: ADVISORY" in out  # followed by its own next iteration only


def test_conditional_drain_covers_nothing(tmp_path):
    r = repo(tmp_path)
    h = write(
        r,
        "ttnn/op/kernels/compute/cond.hpp",
        """
        inline void f(bool x, std::uint32_t *p)
        {
            tensix_sync();
            if (x)
            {
                tensix_sync();
            }
            *p = 1;
        }
        inline void g(bool x)
        {
            tensix_sync();
            if (x)
            {
                tensix_sync();
            }
            tensix_sync();
            TTI_NOP;
        }
        inline void h(bool x, std::uint32_t *p)
        {
            if (x)
            {
                tensix_sync();
                return;
            }
            *p = 1;
        }
        """,
    )
    out = run("--repo", r, h).stdout
    assert "cond.hpp:4:" not in out and "cond.hpp:7:" not in out
    assert (
        "cond.hpp:13: REDUNDANT" in out
    )  # the unconditional drain after the block covers
    assert "cond.hpp:16: REDUNDANT" in out
    assert "cond.hpp:18: ADVISORY" in out
    assert "cond.hpp:25: ADVISORY" in out  # the store is on the other path


def test_conditional_break_poisons_coverage(tmp_path):
    r = repo(tmp_path)
    h = write(
        r,
        "ttnn/op/kernels/compute/brk.hpp",
        """
        inline void f(bool x, std::uint32_t *p)
        {
            for (int i = 0; i < 4; i++)
            {
                tensix_sync();
                if (x)
                    break;
                tensix_sync();
            }
            *p = 1;
        }
        inline void g(std::uint32_t *p)
        {
            for (int i = 0; i < 4; i++)
            {
                tensix_sync();
                break;
            }
            *p = 1;
        }
        """,
    )
    assert run("--repo", r, h).stdout == ""


def test_wrapper_function_is_a_drain_for_its_callers(tmp_path):
    r = repo(tmp_path)
    h = write(
        r,
        "ttnn/op/kernels/compute/wrap.hpp",
        """
        inline void settle() { ckernel::tensix_sync(); }
        inline void twice()
        {
            tensix_sync();
            settle();
        }
        """,
    )
    out = run("--repo", r, h).stdout
    assert "wrap.hpp:2:" not in out
    assert "wrap.hpp:5: REDUNDANT" in out
    assert out.count("wrap.hpp:") == 1


def test_architectures_must_agree(tmp_path):
    r = repo(tmp_path)
    write(
        r,
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_helpers.h",
        "inline void helper() { tensix_sync(); }\n",
    )
    write(
        r,
        "tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_helpers.h",
        "inline void helper() { TTI_NOP; }\n",
    )
    k = write(
        r,
        "ttnn/op/kernels/compute/k.cpp",
        """
        void MAIN {
            tensix_sync();
            helper();
            TTI_NOP;
        }
        """,
    )
    assert (
        "k.cpp:3: ADVISORY" in run("--repo", r, k).stdout
    )  # both agree it orders nothing
    write(
        r,
        "tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_helpers.h",
        "inline void helper() { *done = 1; }\n",
    )
    assert run("--repo", r, k).stdout == ""  # Wormhole's helper needs the drain


def test_local_construction_is_the_type_s_constructor(tmp_path):
    r = repo(tmp_path)
    write(
        r,
        "tt_metal/hw/inc/api/compute/cb.h",
        """
        class CircularBuffer {
        public:
            CircularBuffer(std::uint32_t id) : id_(id) {}
            std::uint32_t id_;
        };
        """,
    )
    k = write(
        r,
        "ttnn/op/kernels/compute/k.cpp",
        """
        void MAIN {
            tensix_sync();
            CircularBuffer cb(3);
            std::uint32_t n{4};
            TTI_NOP;
        }
        """,
    )
    assert "k.cpp:3: ADVISORY" in run("--repo", r, k).stdout


def test_a_returned_or_else_call_is_classified(tmp_path):
    r = repo(tmp_path)
    write(
        r,
        "tt_metal/hw/inc/api/compute/common.h",
        """
        FORCE_INLINE std::uint32_t get_arg_val(int i)
        {
            return *((tt_l1_ptr std::uint32_t*)(get_arg_addr(i)));
        }
        """,
    )
    k = write(
        r,
        "ttnn/op/kernels/compute/k.cpp",
        """
        std::uint32_t f()
        {
            tensix_sync();
            return get_arg_val(0);
        }
        void g(bool x)
        {
            tensix_sync();
            if (x)
                TTI_NOP;
            else
                get_arg_val(0);
        }
        """,
    )
    assert run("--repo", r, k).stdout == ""


def test_quasar_directories_and_third_party_are_skipped(tmp_path):
    r = repo(tmp_path)
    q = write(
        r,
        "tt_metal/hw/ckernels/quasar/metal/x.h",
        "inline void q() { mop_sync(); tensix_sync(); }\n",
    )
    out = run("--repo", r, q).stdout
    assert "REDUNDANT" not in out and "Quasar" in out
    t = write(
        r,
        "tt_metal/third_party/lib/x.h",
        "inline void q() { mop_sync(); tensix_sync(); }\n",
    )
    assert run("--repo", r, t).stdout == ""


def test_all_scans_the_repository_and_skips_files_without_a_drain(tmp_path):
    r = repo(tmp_path)
    write(
        r,
        "ttnn/op/kernels/compute/k.cpp",
        "void MAIN {\n    tensix_sync();\n    TTI_NOP;\n}\n",
    )
    write(
        r, "ttnn/op/kernels/compute/broken.cpp", "void f( {{{{ not parsed, no drain\n"
    )
    out = run("--repo", r, "--all")
    assert "k.cpp:2: ADVISORY" in out.stdout
    assert "Traceback" not in out.stderr


def test_real_repository_runs_clean_of_errors():
    """Every drain site in the repository is parsed; what is found is advisory, not asserted."""
    r = run("--all")
    assert r.returncode == 0
    assert "Traceback" not in r.stderr
