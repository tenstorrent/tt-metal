#!/usr/bin/env python3
"""Static G1 blind-spot analyzer for LLK usage (see docs/llk-formal-contract.md §10).

Reproduce on latest main:
  cd <tt-metal repo root>
  git fetch origin main
  git ls-tree -r --name-only origin/main | grep -E 'kernels/.*compute.*\.cpp$' > /tmp/kernel_list.txt
  rm -rf /tmp/mainsnap && mkdir -p /tmp/mainsnap
  git archive origin/main -- $(tr '\n' ' ' < /tmp/kernel_list.txt) | tar -x -C /tmp/mainsnap
  python3 tt_metal/tt-llk/tools/analysis/blindspot.py

Overlays the §6 FSM on each kernel's call sequence and counts G1 (uninit-restore),
the only in-scope by-design blind spot. Reports certainty = 1 - G1/state-changing calls.
"""

import collections
import os
import re

SNAP = "/tmp/mainsnap"
KLIST = "/tmp/kernel_list.txt"


def strip_code(s):
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    s = re.sub(r'"(\\.|[^"\\])*"', '""', s)
    s = re.sub(r"'(\\.|[^'\\])*'", "''", s)
    return s


NAME = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;{}()\n]*>)?\s*\(")
NOISE = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "static_cast",
    "reinterpret_cast",
    "const_cast",
    "dynamic_cast",
    "get_arg_val",
    "get_compile_time_arg_val",
    "constexpr",
    "UNPACK",
    "MATH",
    "PACK",
    "ckernel",
    "main",
    "MAIN",
    "get_common_arg_val",
    "DPRINT",
    "cb_wait_front",
    "cb_pop_front",
    "cb_reserve_back",
    "cb_push_back",
    "cb_wait_all",
    "ASSERT",
    "static_assert",
    "TT_FATAL",
    "uint32_t",
    "int32_t",
    "float",
    "bool",
    "tensix_sync",
    "noc_async_read",
    "noc_async_write",
    "experimental",
}


def api_class(n):
    if n in NOISE:
        return None
    if n.endswith("_uninit"):
        return "uninit"
    if n.startswith("reconfig"):
        return "reconfig"
    if "hw_startup" in n or n.endswith("hw_configure"):
        return "configure"
    if re.search(r"_init(_short|_common|_once)?$", n):
        return "init"
    return "execute"


EXEC_OK = re.compile(
    r"(_tile|_tiles|_block|_rows?|_cols?|_dest|matmul|reduce|tilize|"
    r"untilize|transpose|pack|copy|bcast|cumsum|_sfpu|eltwise)",
    re.I,
)


def norm_root(n):
    for suf in ("_init_short", "_init_common", "_init_once", "_init", "_uninit"):
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    n = re.sub(r"^(llk_math_|llk_pack_|llk_unpack_|llk_)", "", n)
    n = re.sub(r"_(dest|wh|rows?|cols?|block)$", "", n)
    return n or "x"


def first_arg(args):
    depth = 0
    cur = ""
    for ch in args:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "," and depth == 0:
            break
        cur += ch
    return re.sub(r"\s+", "", cur)


def extract(src):
    """(cls, name, root, cb_first_arg) in call order."""
    out = []
    for m in NAME.finditer(src):
        name = m.group(1)
        cls = api_class(name)
        if cls is None:
            continue
        if cls == "execute" and not EXEC_OK.search(name):
            continue
        # balanced-paren arg capture
        i = src.index("(", m.end() - 1)
        depth = 0
        j = i
        while j < len(src):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        args = src[i + 1 : j]
        out.append((cls, name, norm_root(name), first_arg(args)))
    return out


# ---- §6 FSM: is a uninit transition an ERROR (catchable) or silent? ----
def uninit_is_catchable(state_type, state_op, b_root):
    # silent (non-ERROR) only from INITIALIZED[b] or EXECUTED[b]; else FSM objects
    return not (state_op == b_root and state_type in ("INIT", "EXEC"))


def relies_on_restore(ev, u):
    """After uninit at index u: does the code use operand state before re-setting it?
    reconfig/configure first -> re-established (verifiable). init/execute first -> blind.
    """
    for cls, *_ in ev[u + 1 :]:
        if cls in ("configure", "reconfig"):
            return False  # state explicitly re-established
        if cls in ("init", "execute"):
            return True  # relies on the (unverified) restore
    return False  # nothing downstream depends on it


with open(KLIST) as f:
    kernels = [l.strip() for l in f if l.strip()]

tot = collections.Counter()
n_uninit = 0
catchable = 0
true_g1 = 0
not_relied = 0
g1_files = collections.Counter()
recon_kind = collections.Counter()


def recon_partial(n):
    return bool(re.search(r"(srca|srcb|_df_srca|_df_srcb|single_operand)", n))


for rel in kernels:
    p = os.path.join(SNAP, rel)
    if not os.path.exists(p):
        continue
    ev = extract(strip_code(open(p, errors="replace").read()))
    st_type, st_op = "INITIAL", None
    for idx, (cls, name, root, cb) in enumerate(ev):
        if cls == "execute":
            tot["execute"] += 1
            st_type = "EXEC"
            continue
        tot[cls] += 1
        if cls == "configure":
            st_type, st_op = "CONFIGURED", None
        elif cls == "reconfig":
            recon_kind["partial" if recon_partial(name) else "full"] += 1
            st_type, st_op = "RECONFIGURED", None
        elif cls == "init":
            st_type, st_op = "INIT", root
        elif cls == "uninit":
            n_uninit += 1
            if uninit_is_catchable(st_type, st_op, root):
                catchable += 1
            elif relies_on_restore(ev, idx):
                true_g1 += 1
                g1_files[rel] += 1
            else:
                not_relied += 1
            st_type, st_op = "UNINITIALIZED", None

sc = tot["configure"] + tot["init"] + tot["reconfig"] + tot["uninit"]
print(f"corpus: {len(kernels)} compute kernels @ origin/main\n")
print("state-changing calls (denominator):")
for k in ("configure", "init", "reconfig", "uninit"):
    print(f"  {k:10} {tot[k]:5}")
print(f"  {'TOTAL':10} {sc:5}   (execute op calls: {tot['execute']})\n")

print(f"uninit calls: {n_uninit}")
print(f"  FSM-catchable (tool objects, NOT blind):        {catchable}")
print(f"  FSM-silent, restore not relied on (verifiable): {not_relied}")
print(f"  TRUE G1 (silent + downstream relies on restore): {true_g1}\n")

print(f"reconfig split: partial {recon_kind['partial']}  full {recon_kind['full']}\n")

cert = 100 * (sc - true_g1) / max(sc, 1)
print(f"CERTAINTY (definite verdict on state-changing calls):")
print(f"    {cert:.2f}%   ({sc-true_g1}/{sc})   [in-scope blind = G1 only]\n")

if g1_files:
    print("files with TRUE G1:")
    for rel, n in g1_files.most_common():
        print(f"    {n}  {rel}")
