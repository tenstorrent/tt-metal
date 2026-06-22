# NOP injection plugin for pytest.
#
# Per selected test: run baseline; if it passes, snapshot the ELFs and run a set of
# experiments. Each experiment restores clean ELFs, patches them with ttnop, forces a
# reload, and re-invokes the test so its own golden decides PASS/FAIL.
#
# LLK_PLAN_MODE selects how delays are placed:
# - single (default): Inject one LLK_NOPS delay before a store/load per run.
#     Runs for every site across the chosen threads, one at a time.
# - magnitude: Same as single, but try every delay size in LLK_NSET at each
#     site rather than the single LLK_NOPS value.
# - uniform: Retimes the whole thread by delaying every LLK_CLASSES site,
#     at each magnitude in LLK_NSET.
# - cross: Everything uniform does, but retime several threads at once.
# - skew: Delay one thread heavily and the other lightly.
#
# Env: LLK_NOP (enable), LLK_CLASSES (store,load), LLK_ELFS (unpack,math,pack),
#      LLK_NOPS (2000), LLK_NSET (8,32), LLK_BRANCH_NOPS (200), LLK_SKEW,
#      LLK_LOOP, LLK_MAX_VARIANTS (1), LLK_OUTDIR, TTNOP.

import os
import shutil
import subprocess

import pytest

TTNOP = os.environ.get("TTNOP", "ttnop")  # ttnop binary path
_CFG = {}
_MOD_COUNT = {}


def _sites(elf_path, classes):
    out = subprocess.run(
        [TTNOP, elf_path, "--list"], capture_output=True, text=True
    ).stdout
    sites = []
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 3 and p[0].startswith("0x") and p[2] in classes:
            sites.append((p[0].rstrip(":"), p[2]))
    return sites


def _log(item, text):
    outdir = os.environ.get("LLK_OUTDIR", "/tmp/llk_nop/all")
    os.makedirs(outdir, exist_ok=True)
    name = item.nodeid.replace("/", "_").replace("::", "__")[:170]
    with open(f"{outdir}/{name}.txt", "a", buffering=1) as f:
        f.write(text)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if "LLK_NOP" not in os.environ:
        yield
        return
    from helpers.test_config import TestConfig

    mod = getattr(item, "module", None)
    mod_id = getattr(mod, "__name__", item.nodeid.split("::")[0])

    _CFG.clear()
    orig_run, orig_ref = TestConfig.run, TestConfig.run_elf_files

    def cap(orig):
        def wrapped(self, *a, **k):
            _CFG.setdefault("cfg", self)
            return orig(self, *a, **k)

        return wrapped

    TestConfig.run = cap(orig_run)
    TestConfig.run_elf_files = cap(orig_ref)
    try:
        outcome = yield  # baseline
    finally:
        TestConfig.run = orig_run
        TestConfig.run_elf_files = orig_ref

    max_var = int(os.environ.get("LLK_MAX_VARIANTS", "1"))
    if _MOD_COUNT.get(mod_id, 0) >= max_var:
        return
    if outcome.excinfo is not None:
        _log(item, "# baseline didn't pass, skipped\n")
        return
    if "cfg" not in _CFG:
        _log(item, "# no TestConfig captured, skipped\n")
        return
    _MOD_COUNT[mod_id] = _MOD_COUNT.get(mod_id, 0) + 1
    _run(item)


def _experiments(have, bk):
    """Return list of (label, [(elf, [ttnop-args...]), ...]) per LLK_PLAN_MODE."""
    mode = os.environ.get("LLK_PLAN_MODE", "single")
    classes = os.environ.get("LLK_CLASSES", "store,load").split(",")
    elfs = [
        e
        for e in os.environ.get("LLK_ELFS", "unpack,math,pack").split(",")
        if have.get(e)
    ]
    nset = [int(x) for x in os.environ.get("LLK_NSET", "8,32").split(",")]
    bnops = int(os.environ.get("LLK_BRANCH_NOPS", "200"))
    exps = []

    if mode in ("single", "magnitude"):
        ns = [int(os.environ.get("LLK_NOPS", "2000"))] if mode == "single" else nset
        for elf in elfs:
            for addr, cls in _sites(f"{bk}/{elf}.elf", classes):
                for n in ns:
                    nn = bnops if cls == "branch" else n
                    exps.append((f"{elf}:{addr}={nn}:{cls}", [(elf, [f"{addr}={nn}"])]))
        return exps

    # uniform / cross: globally retime whole threads via --every
    def every(n):
        a = []
        for c in classes:
            a += ["--every", f"{c}={n}"]
        return a

    for elf in elfs:
        for n in nset:
            exps.append((f"uniform:{elf}:every={n}", [(elf, every(n))]))
    if mode == "cross":
        trisc = [e for e in ("unpack", "math", "pack") if have.get(e)]
        for n in nset:
            if len(trisc) == 3:
                exps.append(
                    (f"cross:all3TRISC:every={n}", [(e, every(n)) for e in trisc])
                )
            for a, b in (("unpack", "math"), ("math", "pack"), ("unpack", "pack")):
                if have.get(a) and have.get(b):
                    exps.append(
                        (f"cross:{a}+{b}:every={n}", [(a, every(n)), (b, every(n))])
                    )

    if mode == "skew":
        # Asymmetric per-thread magnitudes: in one run, delay thread A heavily and
        # thread B lightly (and vice versa).
        pairs = [
            ("unpack", "math"),
            ("math", "pack"),
            ("unpack", "pack"),
        ]
        skews = os.environ.get("LLK_SKEW", "500:0,0:500,500:50,50:500").split(",")
        for a, b in pairs:
            if not (have.get(a) and have.get(b)):
                continue
            for sp in skews:
                hi, lo = (int(x) for x in sp.split(":"))
                patches = []
                if hi:
                    patches.append((a, every(hi)))
                if lo:
                    patches.append((b, every(lo)))
                if patches:
                    exps.append((f"skew:{a}={hi}/{b}={lo}", patches))
    return exps


def _run(item):
    from helpers.test_config import TestConfig

    cfg = _CFG["cfg"]
    vdir = str(TestConfig.ARTEFACTS_DIR / cfg.test_name / cfg.variant_id / "elf")
    bk = f"/tmp/llk_nop/bk_generic/{cfg.variant_id}"
    os.makedirs(bk, exist_ok=True)
    have = {}
    for e in ("math", "unpack", "pack"):
        if os.path.exists(f"{vdir}/{e}.elf"):
            shutil.copy(f"{vdir}/{e}.elf", f"{bk}/{e}.elf")
            have[e] = True

    def restore():
        for e in ("math", "unpack", "pack"):
            if have.get(e):
                shutil.copy(f"{bk}/{e}.elf", f"{vdir}/{e}.elf")

    exps = _experiments(have, bk)
    loop = ["--loop"] if os.environ.get("LLK_LOOP") else []
    npass = nmis = nto = nerr = 0
    for label, patches in exps:
        restore()
        ok = True
        for elf, args in patches:
            r = subprocess.run(
                [TTNOP, f"{bk}/{elf}.elf", "-o", f"{vdir}/{elf}.elf", *loop, *args],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                ok = False
                break
        if not ok:
            nerr += 1
            continue
        TestConfig.LAST_LOADED_ELFS = None  # force reload
        try:
            item.runtest()
            npass += 1
        except Exception as ex:  # noqa: BLE001
            s = str(ex)
            if "Timeout" in s or "TIMED OUT" in s:
                nto += 1
                tag = "FAIL-TIMEOUT"
            else:
                nmis += 1
                tag = "FAIL-MISMATCH"
            _log(item, f"{label}\t{tag}\t{s[:80]}\n")
    restore()
    TestConfig.LAST_LOADED_ELFS = None
    _log(
        item,
        f"# SUMMARY mode={os.environ.get('LLK_PLAN_MODE','single')} exps={len(exps)} "
        f"pass={npass} mismatch={nmis} timeout={nto} err={nerr} "
        f"variant={cfg.variant_id[:12]}\n",
    )
