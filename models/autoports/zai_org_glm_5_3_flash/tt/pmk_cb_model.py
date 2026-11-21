# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HOST CB MODEL of a PMK program (W7 / M-B2 design gate, coordinator 2026-10-25 13:40:49Z): the rule behind two silicon
hangs -- ONE producer RISC and ONE consumer RISC per CB index, per core role, inside one window of the program:
  * cb_wait_front snapshots the acked count once: a CB with two consumer RISCs reads stale counts (DFLASH2 rung 3,
    2026-10-23 02:12Z: CB_OUT waited by BRISC and NCRISC -> NCRISC read the RMS / CONV tiles as K);
  * a producer's local push mirror misses another RISC's pushes (14b decode mHC in-program hang, 2026-10-23 03:45Z:
    CB_X / CB_X32 pushed by P1's BRISC stream word AND P6's packer -> X instead of 2X -> the unpack starved).
A phase window re-lays the CB interfaces of every RISC at its barrier (pmk_window.hpp), so the rule is checked per window.

Inputs: a plan engine (tt/pmk_layer.py PmkSparseKdaPlan: phases, roles, windows, CBUse tables) with its bodies registered
(PhaseBody.subparts: the kernel source + compile-time args + defines of every (role, RISC, phase) part).
Method: a STATIC scan of every body source (and its model-local includes): the CB API calls (cb_reserve_back /
cb_push_back = produce, cb_wait_front / cb_pop_front = consume) with their CB argument resolved through the body's own
`constexpr X = get_compile_time_arg_val(PH_CT0 + k)` (k a literal or a header constant such as pfs::CT_X), literal /
macro definitions and the part's defines. RISC of a call: dm0 = BRISC, dm1 = NCRISC; in a compute part the consume calls
run on TRISC0 (unpack) and the produce calls on TRISC2 (pack). CBUse.remote_writers adds REMOTE producers (other cores'
NoC writes + counter increments). The scan over-approximates: calls under `if constexpr` / `#if` branches all count
(the report lists each site so a reviewer can rule a branch out); a CB argument it cannot resolve (a helper's parameter)
is reported as UNRESOLVED with its file / line, never silently dropped.
Output: model(engine) -> dict(per (role, window, cb): producers / consumers / declared format; violations; unresolved);
report_text(m) -> a printable table."""
import os
import re
from collections import defaultdict

PRODUCE = ("cb_reserve_back", "cb_push_back")
CONSUME = ("cb_wait_front", "cb_pop_front")
_CALL = re.compile(r"\b(cb_reserve_back|cb_push_back|cb_wait_front|cb_pop_front)\s*\(")


def _first_arg(text: str, i: int) -> str:
    """the first argument of the call whose '(' ends at text[i - 1] (balanced parentheses / braces)"""
    lvl, out = 0, []
    while i < len(text):
        ch = text[i]
        if ch in "([{":
            lvl += 1
        elif ch in ")]}":
            if lvl == 0:
                break
            lvl -= 1
        elif ch == "," and lvl == 0:
            break
        out.append(ch)
        i += 1
    return "".join(out).strip()


_INC = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
_FUNC = re.compile(r"\b(\w+)\s*\([^;{}()]*(?:\([^;{}()]*\)[^;{}()]*)*\)\s*(?:const\s*)?\{")
_IDENT = re.compile(r"[A-Za-z_]\w*(?:::\w+)*")


# FLAG-EXCLUSIVE pairs (reviewed by hand; a violation whose every site is in one group's files is reported as explained,
# not as a violation): two RISCs act on the CB under complementary RUNTIME flags, so one RISC per configuration
EXCLUSIVE = (
    dict(
        files=("mhc_stream.cpp", "mhc_sync.cpp"),
        why="pmk_mhc: the stream RISC (BRISC) pops the hn OUT tiles / pushes Y only under F_XH_DRAM / F_Y_DRAM (DRAM "
        "mode), the sync RISC (NCRISC) only when the flag is clear (L1 mode): mhc_stream.write_hn / read_y vs "
        "mhc_sync's `if (!(flags & F_XH_DRAM))` / post()'s `if (flags & F_Y_DRAM) return`",
    ),
)


def _risc(part_risc: str, op: str) -> str:
    if part_risc == "dm0":
        return "BRISC"
    if part_risc == "dm1":
        return "NCRISC"
    return "TRISC2.pack" if op in PRODUCE else "TRISC0.unpack"


def _strip(txt: str) -> str:
    """comments out, line structure kept; backslash-continued lines joined (their line count kept by padding)"""
    txt = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), txt, flags=re.S)
    txt = re.sub(r"//[^\n]*", "", txt)
    return re.sub(r"\\\n", " ", txt)


def _int(v):
    try:
        return int(str(v).rstrip("uUlL"), 0)
    except (TypeError, ValueError):
        return None


class _Pre:
    """a minimal C preprocessor over the model's sources (the API headers are not read): #include "..." of model headers
    (each header once per part, as #pragma once / guards make it), #define / #undef (object-like and function-like),
    #if / #ifdef / #ifndef / #elif / #else / #endif (defined(), !, &&, ||, comparisons, integer arithmetic; a condition it
    cannot decide keeps the branch: over-approximation). Output: the kept lines with (file, line); the macro table."""

    def __init__(self, roots, defines):
        self.roots = roots
        self.macros = {str(k): (None, str(v)) for k, v in defines}
        self.lines = []  # (text, file, line)
        self.seen = set()

    def expand(self, expr: str, depth: int = 0) -> str:
        if depth > 12:
            return expr
        out, i = [], 0
        for m in _IDENT.finditer(expr):
            name = m.group(0)
            out.append(expr[i : m.start()])
            i = m.end()
            mac = self.macros.get(name)
            if mac is None:
                out.append(name)
                continue
            params, body = mac
            if params is None:
                out.append("(" + self.expand(body, depth + 1) + ")")
                continue
            j = i  # a function-like macro: its argument list
            while j < len(expr) and expr[j].isspace():
                j += 1
            if j >= len(expr) or expr[j] != "(":
                out.append(name)
                continue
            lvl, k, args, cur = 0, j, [], ""
            while k < len(expr):
                ch = expr[k]
                if ch == "(":
                    lvl += 1
                    if lvl > 1:
                        cur += ch
                elif ch == ")":
                    lvl -= 1
                    if lvl == 0:
                        args.append(cur)
                        break
                    cur += ch
                elif ch == "," and lvl == 1:
                    args.append(cur)
                    cur = ""
                else:
                    cur += ch
                k += 1
            sub = body
            for pn, av in zip(params, args):
                sub = re.sub(rf"\b{re.escape(pn)}\b", "(" + av.strip() + ")", sub)
            out.append("(" + self.expand(sub, depth + 1) + ")")
            i = k + 1
        out.append(expr[i:])
        return "".join(out)

    def cond(self, c: str):
        c = re.sub(r"\bdefined\s*\(\s*(\w+)\s*\)", lambda m: "1" if m.group(1) in self.macros else "0", c)
        c = re.sub(r"\bdefined\s+(\w+)", lambda m: "1" if m.group(1) in self.macros else "0", c)
        c = self.expand(c)
        c = c.replace("&&", " and ").replace("||", " or ")
        c = re.sub(r"!(?!=)", " not ", c)
        c = re.sub(r"\b(\d+)[uUlL]+\b", r"\1", c)
        if re.search(r"[A-Za-z_]\w*", re.sub(r"\b(and|or|not)\b", "", c)):
            return None  # an unknown identifier: undecidable
        try:
            return bool(eval(c, {"__builtins__": {}}, {}))  # noqa: S307 (integer expressions of the model's macros)
        except Exception:  # noqa: BLE001
            return None

    def run(self, path: str, depth: int = 0):
        if path in self.seen or depth > 8:
            return
        self.seen.add(path)
        stack = []  # (active, any_taken)
        active = True
        for n, ln in enumerate(_strip(open(path).read()).split("\n"), 1):
            t = ln.strip()
            if t.startswith("#"):
                d = t[1:].strip()
                kw = d.split(None, 1)[0] if d else ""
                rest = d[len(kw) :].strip()
                if kw in ("if", "ifdef", "ifndef"):
                    if not active:
                        stack.append((active, True, False))
                        active = False
                        continue
                    c = (
                        (rest.split()[0] in self.macros)
                        if kw == "ifdef"
                        else (rest.split()[0] not in self.macros)
                        if kw == "ifndef"
                        else self.cond(rest)
                    )
                    stack.append((active, c is not False, c is None))
                    active = c is not False
                    continue
                if kw == "elif":
                    par, taken, unk = stack[-1]
                    c = self.cond(rest) if par else False
                    if taken and not unk:
                        active = False
                    else:
                        active = par and c is not False
                    stack[-1] = (par, taken or c is True, unk or c is None)
                    continue
                if kw == "else":
                    par, taken, unk = stack[-1]
                    active = par and (not taken or unk)
                    continue
                if kw == "endif":
                    active = stack.pop()[0] if stack else True
                    continue
                if not active:
                    continue
                if kw == "define":
                    m = re.match(r"(\w+)(\(([^)]*)\))?\s*(.*)", rest)
                    if m:
                        params = [x.strip() for x in m.group(3).split(",")] if m.group(2) is not None else None
                        self.macros[m.group(1)] = (params, m.group(4).strip())
                    continue
                if kw == "undef":
                    self.macros.pop(rest.split()[0] if rest else "", None)
                    continue
                if kw == "include":
                    m = _INC.match(t)
                    if m:
                        inc = m.group(1)
                        if inc.startswith(("api/", "tt_metal/", "hostdevcommon/")) or "debug/" in inc:
                            continue
                        cand = [os.path.join(os.path.dirname(path), inc)] + [os.path.join(r, inc) for r in self.roots]
                        p_ = next((c_ for c_ in cand if os.path.exists(c_)), None)
                        if p_ is not None:
                            self.run(p_, depth + 1)
                    continue
                continue
            if active:
                self.lines.append((ln, path, n))


def _match(text: str, i: int, op: str, cl: str) -> int:
    """index just past the bracket closing the one at text[i] (== op), or -1"""
    d = 0
    for j in range(i, len(text)):
        if text[j] == op:
            d += 1
        elif text[j] == cl:
            d -= 1
            if d == 0:
                return j + 1
    return -1


def _constexpr_dead(text: str, val) -> list:
    """(start, end) text ranges of `if constexpr` branches the compile-time args rule out (v6: the PfStream member reader
    pushes CB_X only `if constexpr (XSRC == 0)`, the writer pops CB_OUT only for EPI 0 / 2 -- counting both branches gave
    P13's ACT two producers / three consumers, W7 step-3 mock 16:11Z). A chain whose condition does not resolve keeps
    every branch from there on."""
    dead, seen = [], set()
    for m in re.finditer(r"\bif\s+constexpr\s*\(", text):
        if m.start() in seen:
            continue
        i, taken, ok = m.start(), False, True
        while ok:
            seen.add(i)
            p0 = text.index("(", i)
            p1 = _match(text, p0, "(", ")")
            if p1 < 0:
                break
            cond = text[p0 + 1 : p1 - 1]
            cond = re.sub(r"&&", " and ", re.sub(r"\|\|", " or ", re.sub(r"!(?!=)", " not ", cond)))
            cond = re.sub(r"\btrue\b", "1", re.sub(r"\bfalse\b", "0", cond))
            b0 = p1 + len(text[p1:]) - len(text[p1:].lstrip())
            if b0 >= len(text) or text[b0] != "{":
                break
            b1 = _match(text, b0, "{", "}")
            if b1 < 0:
                break
            v = None if taken else val(cond)
            if taken:
                dead.append((b0, b1))
            elif v is None:
                break  # unresolved: keep this branch and the rest of the chain
            elif v:
                taken = True
            else:
                dead.append((b0, b1))
            rest = text[b1:]
            e0 = b1 + len(rest) - len(rest.lstrip())
            if not text.startswith("else", e0):
                break
            k = e0 + 4
            k += len(text[k:]) - len(text[k:].lstrip())
            if re.match(r"if\s+constexpr\s*\(", text[k:]):
                i = k
                continue
            if k < len(text) and text[k] == "{":  # the final else
                k1 = _match(text, k, "{", "}")
                if k1 > 0 and taken:
                    dead.append((k, k1))
            break
    return dead


def scan_part(src: str, ct: list, defines: list, roots: list, compute: bool = False) -> tuple:
    """one (role, RISC, phase) part -> (found [(op, cb, line, file, function)], unresolved [(op, expr, line, file,
    function)], touched {cb}: the CBs a compute part hands to helpers as B(CB_x) / Buf{CB_x} (op unknown))"""
    pre = _Pre(roots, defines)
    pre.run(src)
    text = "\n".join(t for t, _, _ in pre.lines)
    starts, pos = [], 0
    for t, f, n in pre.lines:
        starts.append((pos, f, n))
        pos += len(t) + 1

    def where(off):
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid][0] <= off:
                lo = mid
            else:
                hi = mid - 1
        return starts[lo][1], starts[lo][2]

    defs, arrs = {}, {}
    for m in re.finditer(r"constexpr\s+(?:static\s+)?[\w:<>]+\s+(\w+)\s*\[\s*\w*\s*\]\s*=\s*\{([^}]*)\}", text):
        arrs[m.group(1)] = m.group(2)
    for m in re.finditer(r"constexpr\s+(?:static\s+)?[\w:<>]+\s+(\w+)\s*=\s*([^;{]+);", text):
        defs.setdefault(m.group(1), m.group(2))  # the first definition in composite order (headers before the body)

    memo, arr_memo = {}, {}

    def arr(nm):
        if nm not in arr_memo:
            arr_memo[nm] = None  # (a self-reference resolves to None)
            lst = pre.expand(arrs[nm]).strip()
            while lst.startswith("(") and lst.endswith(")"):  # an object-like macro's list comes back parenthesised
                lst = lst[1:-1].strip()
            arr_memo[nm] = [val(x.strip().strip("()")) for x in lst.split(",") if x.strip()]
        return arr_memo[nm]

    def val(expr, depth=0):
        key = expr.strip()
        if key in memo:
            return memo[key]
        if depth > 10:
            return None
        memo[key] = None  # cycle guard
        e = pre.expand(key)
        e = re.sub(r"get_compile_time_arg_val\s*\(", "__CT(", e)
        e = e.replace("PH_CT0", "0")
        used = set()

        def ident(m):
            nm = m.group(0).split("::")[-1]
            if nm in ("__CT", "and", "or", "not"):
                return nm
            iv = _int(nm)
            if iv is not None:
                return str(iv)
            if nm in arrs:
                used.add(nm)
                return "__A_" + nm
            if nm in defs and defs[nm].strip() != key:
                v = val(defs[nm], depth + 1)
                return str(v) if v is not None else nm
            return nm

        e2 = _IDENT.sub(ident, e)
        e2 = re.sub(r"\b(\d+)[uUlL]+\b", r"\1", e2)
        if re.search(r"[A-Za-z_]\w*", re.sub(r"\b(__CT|__A_\w+)\b", "", e2)):
            return None
        env = {"__CT": lambda k: int(ct[k]) if 0 <= k < len(ct) else None}
        for nm in used:
            env["__A_" + nm] = arr(nm)
        try:
            v = eval(e2, {"__builtins__": {}}, env)  # noqa: S307 (integer expressions of resolved model constants)
            v = int(v) if v is not None else None
        except Exception:  # noqa: BLE001
            v = None
        memo[key] = v
        return v

    dead = _constexpr_dead(text, val)
    funcs = [
        (m.start(), m.group(1)) for m in _FUNC.finditer(text) if m.group(1) not in ("if", "for", "while", "switch")
    ]
    found, unres = [], []
    for m in _CALL.finditer(text):
        if any(a <= m.start() < b for a, b in dead):  # a branch of an `if constexpr` chain the CT args rule out
            continue
        f, n = where(m.start())
        fn = next((nm for st, nm in reversed(funcs) if st < m.start()), "?")
        arg = _first_arg(text, m.end())
        v = val(arg)
        (found if v is not None else unres).append((m.group(1), v if v is not None else arg, n, f, fn))
    touched = set()
    if compute:  # a compute part's helpers take the CB as a parameter (o, o.cb, cbx): every CB symbol it resolves is
        # treated as touched by compute (unpack and / or pack), op unknown
        body = re.sub(
            r"constexpr\s+(?:static\s+)?[\w:<>]+\s+\w+\s*(?:\[[^\]]*\])?\s*=\s*[^;]+;", " ", text
        )  # definitions out
        for nm in set(re.findall(r"\bCB_\w+", body)):  # only the CB names the compute code actually USES
            v = val(nm)
            if v is not None:
                touched.add(v)
    return found, unres, touched


def model(engine, roles=None, rank: int = 0, scope=None) -> dict:
    """the CB model of ONE program of the engine (scope = its phase ids; None = the engine's current scope): per (role,
    window): cb -> producers / consumers; the violations of the one-producer / one-consumer rule; the unresolved sites
    """
    old_scope = engine.scope
    if scope is not None:
        engine.scope = set(scope)
    try:
        return _model(engine, roles, rank)
    finally:
        engine.scope = old_scope


def _model(engine, roles, rank):
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PL  # noqa: E402

    roots = [PL.REPO_ROOT] if hasattr(PL, "REPO_ROOT") else []
    roles = roles or sorted(engine.HOSTS)  # the COMPOSITE roles (role_cores also lists sub-roles: kda_scan, streamer)
    table = defaultdict(lambda: dict(prod=set(), cons=set(), sites=[], touch=set(), remote=set()))
    kdefs = (
        {}
    )  # (role, risc) -> the composite-wide kernel defines (every registered part's kdefines: kda index maps ..)
    for role in roles:
        for risc in ("dm0", "dm1", "c"):
            kd = {}
            for p in engine.phases:
                if not (engine._hosts(role, risc, p) and engine.in_scope(p.pid)) or p.body.status != "body":
                    continue
                for sub in p.body.subparts(role, risc, rank):
                    for k, v in sub.get("kdefines") or []:
                        kd[str(k)] = str(v)
            kdefs[(role, risc)] = [("PMK_ROLE_" + role.upper(), "1"), ("PMK_REV", "2"), ("MK_PHASED", "1")] + sorted(
                kd.items()
            )
    unresolved, helpers = [], []
    for role in roles:
        wins = engine.windows(role) if hasattr(engine, "windows") else None
        for risc in ("dm0", "dm1", "c"):
            for p in engine.phases:
                if not (engine._hosts(role, risc, p) and engine.in_scope(p.pid)) or p.body.status != "body":
                    continue
                w = engine._win_index(p.pid, wins) if wins else 0
                w = 0 if w is None else w
                for sub in p.body.subparts(role, risc, rank):
                    src = sub.get("src")
                    if not (src and os.path.exists(src)):
                        continue
                    found, unres, touched = scan_part(src, sub.get("ct", []), list(kdefs.get((role, risc), [])) + list(sub.get("defines", [])),
                                                      roots, compute=risc == "c")  # fmt: skip
                    for cb in touched:  # a compute helper's Buf: unpack and / or pack (op unknown)
                        table[(role, w, cb)]["touch"].add("TRISC")
                    for op, cb, line, f, fn in found:
                        e = table[(role, w, cb)]
                        e["prod" if op in PRODUCE else "cons"].add(_risc(risc, op))
                        e["sites"].append((p.pid, sub.get("name", ""), risc, op, os.path.basename(f), line, fn))
                    for op, expr, line, f, fn in unres:
                        if (
                            risc == "c"
                            and re.fullmatch(r"[A-Za-z_]\w*(?:\.\w+)?", str(expr))
                            and not str(expr).startswith("CB_")
                        ):
                            helpers.append(dict(role=role, phase=p.pid, file=os.path.basename(f), line=line, expr=expr))
                            continue  # a compute helper's CB parameter: covered by `touched`
                        unresolved.append(dict(role=role, window=w, phase=p.pid, part=sub.get("name", ""), risc=risc,
                                               op=op, expr=expr, file=os.path.basename(f), line=line, function=fn))  # fmt: skip
        for idx, use in (engine.cbs.get(role) or {}).items():  # REMOTE producers: other cores write + count into it
            for pid in getattr(use, "remote_writers", ()) or ():
                w = engine._win_index(pid, wins) if wins else 0
                table[(role, 0 if w is None else w, idx)]["remote"].add(pid)
    violations, explained = [], []
    for (role, w, cb), e in sorted(table.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if e["remote"] and not e["prod"]:  # other cores write AND count into it (no local producer): REMOTE produces
            e["prod"].add("REMOTE")
        if e["touch"] and e["prod"] - {"TRISC2.pack"} and e["cons"] - {"TRISC0.unpack"}:  # a dm RISC produces AND one
            e["cons"].add("TRISC?")  # consumes while compute also touches it: flag for review
        for kind in ("prod", "cons"):
            if len(e[kind]) > 1:
                files = {s_[4] for s_ in e["sites"] if (s_[3] in PRODUCE) == (kind == "prod")}
                why = next(
                    (g["why"] for g in EXCLUSIVE if files and files <= set(g["files"]) and cb in g.get("cbs", (cb,))),
                    None,
                )
                if why is not None:
                    explained.append(dict(role=role, window=w, cb=cb, kind=kind, riscs=sorted(e[kind]), why=why))
                    continue
                violations.append(dict(role=role, window=w, cb=cb, kind=kind, riscs=sorted(e[kind]),
                                       sites=[s for s in e["sites"] if (s[3] in PRODUCE) == (kind == "prod")]))  # fmt: skip
    reclaim = _reclaim_hazards(engine, roles)
    cbs = {f"{r}/w{w}/cb{c}": dict(prod=sorted(e["prod"]), cons=sorted(e["cons"]), touch=sorted(e["touch"]),
                                    remote=sorted(e["remote"]), declared=_declared(engine, r, c))
           for (r, w, c), e in table.items()}  # fmt: skip
    return dict(cbs=cbs, violations=violations, explained=explained, unresolved=unresolved, n_cbs=len(table),
                n_violations=len(violations), n_explained=len(explained), n_unresolved=len(unresolved),
                n_helper_sites=len(helpers), reclaim=reclaim,
                n_reclaim_hazards=sum(1 for h in reclaim if not h["covered_by"]))  # fmt: skip


def _reclaim_hazards(engine, roles) -> list:
    """THE RECLAIM RULE (coordinator 2026-10-25 15:18:55Z, after the W7 step-1 hang): a CB that OTHER cores write
    (CBUse.remote_writers: lanes, peer chips) in phase q, whose bytes in q's window overlap an allocation of the role's
    RECLAIM window, must be ordered by an explicit wait on the owner BEFORE enter_reclaim: a hand-off q signals, waited
    by the role's data-movement composites in a phase after q and before the reclaim, or listed in the engine's
    reclaim_drain. The core barrier of enter_reclaim sees only the core's own traffic (w7s1w: P11's peer x-hat_f packets
    into XS 35 landed in B's CBs 49 / 59 on device 1). One entry per (role, cb, q); covered_by = [] is a HAZARD."""
    out = []
    order = {p.pid: n for n, p in enumerate(engine.phases)}
    ph = {p.pid: p for p in engine.phases}
    al = lambda b: -(-int(b) // engine.WIN_ALIGN) * engine.WIN_ALIGN  # noqa: E731
    for role in roles:
        rec = (getattr(engine, "win_reclaim", None) or {}).get(role)
        if rec is None or not engine.in_scope(rec["pid"]):
            continue
        wl = engine._win_layout(role)
        if not wl or not wl.get("reclaim"):
            continue
        wins = engine.windows(role)
        # B's reclaim (WIN_RECLAIM, rec["pid"]) is the role's FIRST reclaim window; WIN_RECLAIM2 (P15's RS8 CBs over B's
        # dead region, d3e3b152fe9) follows it -- the single-window unpack raised 'too many values' on every B7g build
        (rw, rb) = sorted(wl["reclaim"].items())[0]
        rtab = [(int(c), int(o), int(n)) for c, o, n, _ in wl["tabs"][rw]]
        pos0, pos = {}, 0  # the launch declaration is contiguous from the CB base = the reclaim region's base
        for us, b in wl["decl"]:
            for u in us:
                pos0.setdefault(int(u.index), (pos, int(b)))
            pos += al(b)
        rp = rec["pid"]
        drains = tuple(engine._drain_hids(role, rp)) if hasattr(engine, "_drain_hids") else ()

        def waited_after(q):
            hs = set()
            for p in engine.phases:
                if not (order[q] < order[p.pid] < order[rp]) or not engine.in_scope(p.pid) or p.body.status != "body":
                    continue
                if role not in engine.wait_roles(p) or not any(engine._hosts(role, r_, p) for r_ in ("dm0", "dm1")):
                    continue
                hs |= {h for h in p.waits if engine._has_live_producer(h)}
            return hs

        for idx, use in sorted((engine.cbs.get(role) or {}).items()):
            for q in getattr(use, "remote_writers", ()) or ():
                if q not in ph or not engine.in_scope(q) or ph[q].body.status != "body":
                    continue
                wq = engine._win_index(q, wins)
                if wq is None or wq >= rw:
                    continue
                ent = next((t for t in wl["tabs"].get(wq, ()) if int(t[0]) == int(idx)), None) if wq else None
                if ent is not None:
                    lo, n = rb + int(ent[1]), int(ent[2])
                elif int(idx) in pos0:  # window 1 in place, or a shared allocation
                    lo, n = pos0[int(idx)]
                else:
                    out.append(dict(role=role, cb=int(idx), phase=q, window=wq, range=None, reclaim_cbs=[],
                                    covered_by=[], why="no placement found for the remote-written CB"))  # fmt: skip
                    continue
                over = sorted({c for c, o, n_ in rtab if o < lo + n and lo < o + n_})
                if not over:
                    continue
                hs = waited_after(q)
                cover = [f"{h} (wait)" for h in ph[q].signals if h in hs] + [
                    f"{h} (drain)" for h in ph[q].signals if h in drains
                ]
                out.append(dict(role=role, cb=int(idx), phase=q, window=wq, range=[lo, lo + n], reclaim_cbs=over,
                                covered_by=cover))  # fmt: skip
    return out


def _declared(engine, role, idx):
    use = (engine.cbs.get(role) or {}).get(idx)
    if use is None:
        return None
    return dict(
        fmt=use.fmt, pages=use.pages, page=use.page, live=list(use.live), solo=bool(getattr(use, "solo", False))
    )


def report_text(m: dict, max_sites: int = 6) -> str:
    out = [f"PMK host CB model: {m['n_cbs']} (role, window, cb) entries, {m['n_violations']} violations of the one-producer / "
           f"one-consumer rule, {m.get('n_explained', 0)} explained (flag-exclusive), {m['n_unresolved']} unresolved call sites "
           f"(+ {m.get('n_helper_sites', 0)} compute-helper sites: CB parameters, counted as compute-touched)"]  # fmt: skip
    for h in m.get("reclaim", []):
        tag = "reclaim ok" if h["covered_by"] else "RECLAIM HAZARD"
        out.append(f"  {tag} {h['role']} cb {h['cb']}: remote writes in {h['phase']} (window {h['window'] + 1}) at reclaim "
                   f"offsets {h['range']} over reclaim CBs {h['reclaim_cbs']}; ordered by {h['covered_by'] or 'NOTHING'}"
                   + (f" ({h['why']})" if h.get("why") else ""))  # fmt: skip
    for v in m.get("explained", []):
        out.append(f"  explained {v['role']} window {v['window']} cb {v['cb']}: {v['kind']} {v['riscs']} -- {v['why']}")
    for v in m["violations"]:
        out.append(
            f"  VIOLATION {v['role']} window {v['window']} cb {v['cb']}: {len(v['riscs'])} {v['kind']} RISCs {v['riscs']}"
        )
        for s in v["sites"][:max_sites]:
            out.append(f"      {s[0]}{('/' + s[1]) if s[1] else ''} {s[2]} {s[3]} at {s[4]}:{s[5]} ({s[6]})")
    seen = set()
    for u in m["unresolved"]:
        k = (u["file"], u["line"])
        if k in seen:
            continue
        seen.add(k)
        out.append(
            f"  UNRESOLVED {u['role']} {u['phase']} {u['risc']} {u['op']}({u['expr']}) at {u['file']}:{u['line']} ({u['function']})"
        )
    return "\n".join(out)
