#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tools over the LLK contract (llk_contract.yaml, with llk_bindings.yaml next to it): every
entry point described on its own.

  validate    the file follows the schema; every field named exists and every index is an operand or
              an instance of the field; an entry point that changes one field of an invariant changes
              the others. Then four properties of the contracts, none a rule on call order:
                establish   every field an execute requires is ensured, with the same value, by its own
                            operation's init, by configure, by the execute itself, or at kernel start
                restore     what an init changes away from configure's value (or kernel start's), and
                            whether its uninit ensures that value again
                interfere   A leaves a field changed that B's execute takes from configure (or kernel start)
                between     B's execute takes a field from B's init; entry point A changes it, so A
                            between B's init and B's execute needs B's init again (configure is left
                            out: it runs once, at kernel start)
  check-code  against a tt-metal checkout: the LLK functions named for an entry point set every field
              it ensures; and what they set that the entry point does not mention. What CI would run.
  check-api   the compute API functions listed for each entry point (1.0 and 2.0) reach its LLK functions
  trace       every requirement, by ID, with its reason and, for what an entry point ensures, the lines
              of LLK code that set it: requirement <-> code line
  check-sequence  a sequence of entry-point calls, checked against the contract alone (no chip): before
              each call, is what it requires there? After it, what it ensures holds, what it may change
              is unknown, and everything else is as it was. Reports each requirement not met, by ID,
              with the call that last set the field

Every requires/ensures item has an ID made from where it is: <operation>.<entry>.<requires|ensures>:
<field>, e.g. tilize.execute.requires:unpack.Tileize_mode. An entry point may give a reason for an
item in `why`.

Values are compared as text after each operand name (in0, in1) is replaced by the unpacker that
carries it in that operation (srcA, srcB), so that configure's in0 and matmul's in1 compare equal.

Usage, from the folder above tools/:
  python3 tools/contract_tool.py validate       contract/llk_contract.yaml      (reads the schema next to it)
  python3 tools/contract_tool.py check-code     contract/llk_contract.yaml ~/tt-metal
  python3 tools/contract_tool.py check-api      contract/llk_contract.yaml api_map.json   (from compute_api_map.py --all)
  python3 tools/contract_tool.py trace          contract/llk_contract.yaml ~/tt-metal [--md out.md]
  python3 tools/contract_tool.py check-sequence contract/llk_contract.yaml examples/sequences/tilize_ok.yaml
"""

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ENTRIES = ("hw_configure", "reconfigure", "init", "execute", "uninit")
UNITS = ("unpack", "math", "pack")
REF = re.compile(r"^(unpack|math|pack)\.(\w+)(?:\[(\w+)\])?$")
OPERANDS = ("in0", "in1", "out")


BINDINGS = "llk_bindings.yaml"
BOUND = ("llk", "compute_api", "template")


def load(path):
    """The contract, with llk_bindings.yaml (next to it) merged into its entry points.
    The files as written are kept in "__files__" for validate."""
    path = Path(path).resolve()
    raw = yaml.safe_load(open(path))
    bpath = path.with_name(BINDINGS)
    bindings = yaml.safe_load(open(bpath)) if bpath.exists() else {}
    c = copy.deepcopy(raw)
    unbound = []
    for key, b in (bindings or {}).items():
        op_name, _, entry = key.partition(".")
        en = c.get("operations", {}).get(op_name, {}).get(entry)
        if en is None:
            unbound.append(key)
            continue
        en.update({k: v for k, v in b.items() if k in BOUND})
    c["__files__"] = {
        "contract": raw,
        "bindings": bindings or {},
        "unbound": unbound,
        "dir": path.parent,
    }
    return c


def entries(c):
    """(op_name, entry_name, op, entry) for every entry point."""
    for name, op in c["operations"].items():
        for e in ENTRIES:
            if e in op:
                yield name, e, op, op[e]


def rid(op_name, entry, part, ref):
    """The ID of a requires/ensures item."""
    return f"{op_name}.{entry}.{part}:{ref}"


def parse_ref(ref):
    m = REF.match(ref)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def instances_of(field):
    if isinstance(field.get("regs"), dict):
        return list(field["regs"])
    return list(field.get("instances", []))


def instance(op, index):
    """The instance an index names in an operation: an operand's unpacker, or the index itself."""
    if index in OPERANDS:
        return op.get("operands", {}).get(index, {}).get("unpacker", index)
    return index


def canon_ref(op, ref):
    u, f, i = parse_ref(ref)
    return f"{u}.{f}" + (f"[{instance(op, i)}]" if i else "")


def canon_val(op, val):
    s = str(val)
    for o in ("in0", "in1"):
        inst = op.get("operands", {}).get(o, {}).get("unpacker")
        if inst:
            s = re.sub(rf"\b{o}\b", inst, s)
    return s


def canon(op, d):
    """A requires/ensures map with references and values named by instance."""
    return {canon_ref(op, r): canon_val(op, v) for r, v in d.items()}


def thread_of(c, ref):
    """The thread that writes a field: its `thread`, or its unit."""
    u, f, _ = parse_ref(ref)
    return c["state"][u][f].get("thread", u)


class Defs:
    """LLK function definitions, specialized per (thread, template arguments) of an entry point."""

    def __init__(self, ldefs):
        self.base, self.cache = ldefs, {}

    def get(self, args):
        from llk_source import specialize

        key = tuple(sorted((args or {}).items()))
        if key not in self.cache:
            if not args:
                d = self.base
            else:
                d = {
                    n: [
                        (rng, params, specialize(body, args), rel)
                        for rng, params, body, rel in v
                    ]
                    for n, v in self.base.items()
                }
            self.cache[key] = (d, {}, {})  # defs, writes memo, bodies memo
        return self.cache[key]


def regs_of(c, op, ref):
    """cfg_defines names behind a field reference, the instance picked by the index."""
    unit, field, index = parse_ref(ref)
    regs = c["state"][unit][field].get("regs", [])
    if isinstance(regs, dict):
        if index:
            regs = regs.get(instance(op, index), [])
        else:
            regs = [
                r for v in regs.values() for r in (v if isinstance(v, list) else [v])
            ]
    return regs if isinstance(regs, list) else [regs]


def detect_of(c, op, ref):
    unit, field, index = parse_ref(ref)
    rx = c["state"][unit][field].get("detect")
    if not rx:
        return None
    if "{n}" in rx:
        n = re.sub(r"\D", "", instance(op, index) or "")
        rx = rx.replace("{n}", n)
    return re.compile(rx)


# --------------------------------------------------------------------------------------------- validate
def validate(c):
    errors, notes = [], []
    files = c.get(
        "__files__", {"contract": c, "bindings": {}, "unbound": [], "dir": HERE}
    )
    for doc, schema in (
        (files["contract"], "llk_contract.schema.yaml"),
        (files["bindings"], "llk_bindings.schema.yaml"),
    ):
        try:
            import jsonschema

            jsonschema.validate(doc, yaml.safe_load(open(Path(files["dir"]) / schema)))
        except ImportError:
            notes.append("jsonschema not installed: schema check skipped")
            break
        except FileNotFoundError:
            notes.append(f"{schema} not found: schema check skipped")
        except Exception as e:
            errors.append(f"{schema}: {getattr(e, 'message', e)}")
    for key in files["unbound"]:
        errors.append(f"{BINDINGS}: {key} is not an entry point of the contract")
    for name, ename, op, en in entries(c):
        if name != "configure" and "llk" not in en:
            notes.append(
                f"{name}.{ename} has no binding in {BINDINGS}: code checks skip it"
            )
    st = c["state"]

    def check_ref(where, op, ref):
        p = parse_ref(ref)
        if not p or p[1] not in st.get(p[0], {}):
            errors.append(f"{where}: {ref} is not a field of `state`")
            return False
        u, f, i = p
        fld = st[u][f]
        insts = instances_of(fld)
        if i is None:
            if fld.get("instances"):
                errors.append(f"{where}: {ref} needs an instance ({', '.join(insts)})")
            return True
        if op is not None and i in OPERANDS:
            if i not in op.get("operands", {}):
                errors.append(
                    f"{where}: {ref} names operand {i}, which the operation does not have"
                )
                return False
        inst = instance(op or {}, i)
        if insts and inst not in insts:
            errors.append(
                f"{where}: {ref}: {inst} is not an instance of {u}.{f} ({', '.join(insts)})"
            )
            return False
        if not insts:
            errors.append(f"{where}: {ref}: {u}.{f} has no instances")
            return False
        return True

    def base(ref):
        u, f, _ = parse_ref(ref)
        return f"{u}.{f}"

    invs = c.get("invariants", [])
    for inv in invs:
        for ref in inv["fields"]:
            check_ref(inv["id"], None, ref)
    for ref in c.get("kernel_start", {}).get("assumes", {}):
        check_ref("kernel_start", None, ref)

    for name, ename, op, en in entries(c):
        where = f"{name}.{ename}"
        refs = (
            list(en.get("requires", {}))
            + list(en.get("ensures", {}))
            + en.get("may_change", [])
        )
        if not all([check_ref(where, op, r) for r in refs]):
            continue
        for ref in en.get("why", {}):
            if ref not in en.get("requires", {}) and ref not in en.get("ensures", {}):
                errors.append(
                    f"{where}: why explains {ref}, which is in neither requires nor ensures"
                )
        touched = {
            base(r) for r in list(en.get("ensures", {})) + en.get("may_change", [])
        }
        for inv in invs:
            fs = {base(r) for r in inv["fields"]}
            if fs & touched and not fs <= touched:
                errors.append(
                    f"{where}: changes {', '.join(sorted(fs & touched))} but not {', '.join(sorted(fs - touched))} ({inv['id']})"
                )
    if errors:
        report(errors, notes, [])
        return errors

    cfg_op = c["operations"].get("configure", {})
    cfg = canon(cfg_op, cfg_op.get("hw_configure", {}).get("ensures", {}))
    start = {r: str(v) for r, v in c.get("kernel_start", {}).get("assumes", {}).items()}
    baseline = {**start, **cfg}
    findings = []

    # establish
    for name, op in c["operations"].items():
        ex = op.get("execute")
        if not ex:
            continue
        init = canon(op, op.get("init", {}).get("ensures", {}))
        own = canon(op, ex.get("ensures", {}))
        for oref, oval in ex.get("requires", {}).items():
            ref, val = canon_ref(op, oref), canon_val(op, oval)
            srcs = (
                ("init", init),
                ("configure", cfg),
                ("kernel start", start),
                ("execute", own),
            )
            by = [(s, d[ref]) for s, d in srcs if ref in d]
            if not by:
                findings.append(
                    (
                        "establish",
                        rid(name, "execute", "requires", oref),
                        f"= {val}; no entry point ensures it",
                    )
                )
            elif by[0][1] != val:
                findings.append(
                    (
                        "establish",
                        rid(name, "execute", "requires", oref),
                        f"= {val}; {by[0][0]} ensures {by[0][1]}",
                    )
                )

    # restore
    leaves = defaultdict(dict)
    for name, op in c["operations"].items():
        init = canon(op, op.get("init", {}).get("ensures", {}))
        uninit = (
            canon(op, op.get("uninit", {}).get("ensures", {}))
            if op.get("uninit")
            else {}
        )
        for ref, val in init.items():
            if ref in baseline and baseline[ref] != val:
                if uninit.get(ref) == baseline[ref]:
                    continue
                leaves[name][ref] = uninit.get(ref, val)
                how = (
                    "no uninit"
                    if not op.get("uninit")
                    else (
                        f"uninit ensures {uninit[ref]}"
                        if ref in uninit
                        else "uninit leaves it"
                    )
                )
                findings.append(
                    (
                        "restore",
                        name,
                        f"init sets {ref} = {val} (configure: {baseline[ref]}); {how}",
                    )
                )

    # interfere
    for b, opb in c["operations"].items():
        ex = opb.get("execute")
        if not ex:
            continue
        own = canon(opb, opb.get("init", {}).get("ensures", {}))
        relies = {
            r
            for r in canon(opb, ex.get("requires", {}))
            if r not in own and r in baseline
        }
        for a, left in leaves.items():
            if a == b:
                continue
            for ref in sorted(relies & set(left)):
                findings.append(
                    (
                        "interfere",
                        f"{a} -> {b}",
                        f"{a} leaves {ref} = {left[ref]}; {b} execute takes configure's {baseline[ref]}",
                    )
                )

    # between
    for b, opb in c["operations"].items():
        ex, ini = opb.get("execute"), opb.get("init")
        if not ex or not ini:
            continue
        init = canon(opb, ini.get("ensures", {}))
        req = canon(opb, ex.get("requires", {}))
        from_init = {r: v for r, v in req.items() if r in init and init[r] == v}
        for a, ea, opa, en in entries(c):
            if a == b or ea == "hw_configure":  # configure runs once, at kernel start
                continue
            ens = canon(opa, en.get("ensures", {}))
            may = {canon_ref(opa, r) for r in en.get("may_change", [])}
            hit = sorted(
                r
                for r in from_init
                if (r in ens and ens[r] != from_init[r]) or r in may
            )
            if hit:
                findings.append(
                    (
                        "between",
                        f"{a}.{ea} | {b}",
                        f"changes {', '.join(hit)}, which {b} execute takes from {b} init",
                    )
                )

    report(errors, notes, findings)
    return errors


def report(errors, notes, findings):
    print(f"validate: {len(errors)} errors, {len(findings)} findings")
    for e in errors:
        print(f"  ERROR    {e}")
    for n in notes:
        print(f"  note     {n}")
    for kind, where, text in findings:
        print(f"  {kind:<9} {where}: {text}")


# ------------------------------------------------------------------------------------------- check-code
WORD_W = re.compile(r"\bcfg\s*\[\s*([A-Z]\w*)_ADDR32\s*(?:\+\s*(\w+))?\s*\]\s*=(?!=)")
WRCFG_W = re.compile(
    r"\bTTI?_WRCFG\s*\(([^;]*?)\b([A-Z]\w*)_ADDR32\s*(?:\+\s*(\w+))?\s*\)"
)


RMW_MASKED = re.compile(
    r"cfg_reg_rmw_tensix\s*<\s*[A-Z]\w*_ADDR32\s*,\s*\d+\s*,\s*([A-Za-z_]\w*)\s*>"
)
RMW_PAIR = re.compile(
    r"cfg_reg_rmw_tensix\s*<\s*([A-Z]\w*)_ADDR32\s*,\s*([A-Z]\w*)_SHAMT"
)


def masked_fields(text):
    """Fields an RMW sets through a named mask (alu_mask = A_MASK | B_MASK ...), followed through
    the constants the mask is built from."""
    out = set()

    def expand(name, seen):
        if name in seen:
            return
        seen.add(name)
        m = re.search(r"\b" + re.escape(name) + r"\s*=\s*([^;]+);", text)
        if not m:
            return
        for f in re.findall(r"\b([A-Z][A-Za-z0-9_]*)_MASK\b", m.group(1)):
            out.add(f)
        for ident in re.findall(r"\b([a-z_]\w*)\b", m.group(1)):
            expand(ident, seen)

    for m in RMW_MASKED.finditer(text):
        expand(m.group(1), set())
    return out


def cfg_table(repo):
    p = repo / "tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h"
    return {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"#define\s+(\w+)_ADDR32\s+(\d+)", p.read_text())
    }


def words_written(text, addr):
    """Config words written whole (cfg[...] = and WRCFG), with literal or loop offsets."""
    out = set()

    def span(base, off, n):
        if base not in addr:
            return
        if off is None:
            k = [0]
        elif off.isdigit():
            k = [int(off)]
        else:
            k = range(4)  # a loop index: one 128-bit config block
        for i in k:
            for j in range(n):
                out.add(addr[base] + i + j)

    for m in WORD_W.finditer(text):
        span(m.group(1), m.group(2), 1)
    for m in WRCFG_W.finditer(text):
        span(m.group(2), m.group(3), 4 if "128b" in m.group(1) else 1)
    return out


def check_code(c, repo):
    sys.path.insert(0, str(HERE))
    import llk_cfg_map as lm
    from llk_source import bodies_of

    defs = Defs(lm.build(repo))
    addr = cfg_table(repo)

    def all_regs(f):
        v = f.get("regs", [])
        vals = v.values() if isinstance(v, dict) else [v]
        return [x for y in vals for x in (y if isinstance(y, list) else [y])]

    for u in c["state"].values():
        for f in u.values():
            for r in all_regs(f):
                if r not in addr:
                    print(f"  UNKNOWN  register {r} is not in cfg_defines.h")

    def facts(fns, en):
        """What the (thread, function) pairs write, with the entry point's template arguments."""
        names, missing, text = set(), [], []
        for t, fn in fns:
            ldefs, wmemo, bmemo = defs.get(en.get("template", {}).get(t))
            if fn not in ldefs:
                missing.append(fn)
                continue
            names |= {reg for reg, how, k, via in lm.writes_of(ldefs, fn, wmemo)}
            text.append(bodies_of(ldefs, fn, bmemo))
        text = "\n".join(text)
        names |= masked_fields(text)
        # cfg_reg_rmw_tensix<A_ADDR32, B_SHAMT, ...> with A and B in different words writes A's word
        names -= {
            b
            for a, b in RMW_PAIR.findall(text)
            if a != b and addr.get(a) != addr.get(b)
        }
        return names, words_written(text, addr), text, missing

    def sets(op, ref, names, words, text):
        rx = detect_of(c, op, ref)
        if rx is not None:
            return bool(rx.search(text))
        regs = regs_of(c, op, ref)
        return any(r in names for r in regs) or any(addr.get(r) in words for r in regs)

    # non-register fields and their instances, to report code that sets one the entry point does not mention
    detect_fields = []
    for u, fs in c["state"].items():
        for f, d in fs.items():
            if d.get("detect"):
                for i in d.get("instances") or [None]:
                    detect_fields.append((u, f"{u}.{f}" + (f"[{i}]" if i else "")))

    total = ok = extra = 0
    for name, ename, op, en in entries(c):
        llk = en.get("llk", {})
        where = f"{name}.{ename}"
        mentioned = list(en.get("ensures", {})) + en.get("may_change", [])
        for ref in en.get("ensures", {}):
            u = thread_of(c, ref)
            fns = (
                list(llk.items())
                if ename == "hw_configure"
                else [(u, llk[u])] if u in llk else []
            )
            total += 1
            if not fns:
                print(
                    f"  MISSING  {where}: ensures {ref}, but names no LLK function on {u}"
                )
                continue
            names, words, text, miss = facts(fns, en)
            if miss:
                print(
                    f"  MISSING  {where}: {', '.join(miss)} not found in the LLK source"
                )
                continue
            if sets(op, ref, names, words, text):
                ok += 1
            else:
                print(
                    f"  MISSING  {rid(name, ename, 'ensures', ref)}: {' + '.join(f for _, f in fns)} does not set it"
                )
        for u, fn in llk.items():
            names, words, text, miss = facts([(u, fn)], en)
            scope = (
                mentioned
                if ename == "hw_configure"
                else [r for r in mentioned if thread_of(c, r) == u]
            )
            known = {
                r
                for ref in scope
                if not detect_of(c, op, ref)
                for r in regs_of(c, op, ref)
            }
            known_words = {addr.get(r) for r in known}
            for reg in sorted(names):
                if reg not in known and addr.get(reg) not in known_words:
                    extra += 1
                    print(
                        f"  EXTRA    {where}: {fn} writes {reg}, which the entry point does not mention"
                    )
            kn = {canon_ref(op, r) for r in scope}
            for fu, ref in detect_fields:
                if (ename != "hw_configure" and thread_of(c, ref) != u) or canon_ref(
                    op, ref
                ) in kn:
                    continue
                if detect_of(c, op, ref).search(text):
                    extra += 1
                    print(
                        f"  EXTRA    {where}: {fn} sets {ref}, which the entry point does not mention"
                    )
    print(
        f"check-code: {ok} of {total} fields the entry points ensure are set by the LLK functions named for them; {extra} writes not mentioned"
    )


# -------------------------------------------------------------------------------------------- check-api
# ------------------------------------------------------------------------------------------------ trace
def write_sites(c, op, ref, body, fulltext, addr):
    """Offsets in `body` where the field `ref` is set. `fulltext` is the caller's text with its
    helpers, where a mask an RMW uses may be defined."""
    import llk_cfg_map as lm

    rx = detect_of(c, op, ref)
    if rx is not None:
        return [m.start() for m in rx.finditer(body)]
    regs = set(regs_of(c, op, ref))
    words = {addr.get(r) for r in regs}
    out = []
    for how, wrx in lm.WRITES:
        for m in wrx.finditer(body):
            names = {m.group(1)}
            if how == "rmw":
                p = lm.RMW_PART.match(body, m.start())
                if p:
                    names = {p.group(1), p.group(2)}
                mm = RMW_MASKED.match(body, m.start())
                if mm:
                    names |= masked_fields_of(mm.group(1), body + "\n" + fulltext)
            hit = bool(names & regs)
            if not hit and how in ("word", "wrcfg"):
                end = body.find(";", m.end())
                hit = bool(
                    words_written(
                        body[m.start() : (len(body) if end < 0 else end + 1)], addr
                    )
                    & words
                )
            if hit:
                out.append(m.start())
    return sorted(set(out))


def masked_fields_of(mask, text):
    out, seen = set(), set()

    def expand(name):
        if name in seen:
            return
        seen.add(name)
        m = re.search(r"\b" + re.escape(name) + r"\s*=\s*([^;]+);", text)
        if not m:
            return
        out.update(re.findall(r"\b([A-Z][A-Za-z0-9_]*)_MASK\b", m.group(1)))
        for ident in re.findall(r"\b([a-z_]\w*)\b", m.group(1)):
            expand(ident)

    expand(mask)
    return out


def trace(c, repo, md_path=None, only_op=None):
    sys.path.insert(0, str(HERE))
    import llk_cfg_map as lm
    import llk_source as sc
    from llk_source import bodies_of

    defs = Defs(lm.build(repo))
    base = defs.base
    addr = cfg_table(repo)
    stripped = {}

    def line_of(rel, body, off):
        if rel not in stripped:
            stripped[rel] = sc.strip((repo / rel).read_text(errors="replace"))
        pos = stripped[rel].find(body)
        return None if pos < 0 else stripped[rel].count("\n", 0, pos + off) + 1

    def sites(op, ref, fn, args):
        """file:line of every place fn, or a helper it calls, sets the field (template arguments applied)."""
        ldefs, _, bmemo = defs.get(args)
        full = bodies_of(ldefs, fn, bmemo)
        out, seen = [], set()

        def walk(name, stack):
            if name in stack or name not in ldefs:
                return
            for (rng, params, body, rel), orig in zip(ldefs[name], base[name]):
                key = (rel, name, len(body))
                if key in seen:
                    continue
                seen.add(key)
                for off in write_sites(c, op, ref, body, full, addr):
                    ln = line_of(rel, orig[2], off)
                    out.append(f"{rel.split('/tt_llk_blackhole/')[-1]}:{ln} ({name})")
                for callee, argc in sc.calls_with_arity(body):
                    if callee in ldefs and callee != name:
                        walk(callee, stack + (name,))

        walk(fn, ())
        return out

    rows, heads = [], []
    for name, ename, op, en in entries(c):
        if only_op and name not in only_op.split(","):
            continue
        heads.append((name, ename, en, len(rows)))
        why = en.get("why", {})
        llk = en.get("llk", {})
        for part in ("requires", "ensures"):
            for ref, val in en.get(part, {}).items():
                u = thread_of(c, ref)
                fns = (
                    list(llk.items())
                    if ename == "hw_configure"
                    else [(u, llk[u])] if u in llk else []
                )
                where = []
                if part == "ensures":
                    for t, fn in fns:
                        where += sites(op, ref, fn, en.get("template", {}).get(t))
                rows.append(
                    (
                        rid(name, ename, part, ref),
                        str(val),
                        why.get(ref, ""),
                        ", ".join(f for _, f in fns),
                        where,
                    )
                )
    n_ens = sum(1 for r in rows if ".ensures:" in r[0])
    n_found = sum(1 for r in rows if ".ensures:" in r[0] and r[4])
    for i, v, w, f, where in rows:
        print(f"{i}\n    value  {v}")
        if w:
            print(f"    why    {w}")
        if ".ensures:" in i:
            print(
                f"    set at {'; '.join(where[:6]) + (' ...' if len(where) > 6 else '') if where else 'NOT FOUND in ' + f}"
            )
        else:
            print(f"    relied on by {f or '(no LLK named on this unit)'}")
    use = state_use(c)
    print("\nstate the entry points rely on (what a runtime checker has to track):")
    for f, by in use["relied"].items():
        print(f"  {f:<34} {', '.join(by)}")
    print("state no entry point relies on (changed, or left with any value):")
    for f, by in use["changed"].items():
        print(f"  {f:<34} {', '.join(by)}")
    if use["unused"]:
        print("state no entry point mentions yet: " + ", ".join(use["unused"]))
    print(
        f"trace: {len(rows)} requirements; {n_found} of {n_ens} ensures traced to code lines; "
        f"{len(use['relied'])} fields relied on, {len(use['changed'])} not"
    )
    if md_path:
        write_trace_md(c, md_path, rows, heads, only_op, None if only_op else use)


def state_use(c):
    """Which fields (by instance) some entry point requires, which are only changed, which are unused.
    Derived from the entry points' lists, so it never disagrees with them."""
    relied, changed = defaultdict(list), defaultdict(list)
    for name, ename, op, en in entries(c):
        for ref in en.get("requires", {}):
            relied[canon_ref(op, ref)].append(f"{name}.{ename}")
        for ref in list(en.get("ensures", {})) + en.get("may_change", []):
            changed[canon_ref(op, ref)].append(f"{name}.{ename}")
    for ref in c.get("kernel_start", {}).get("assumes", {}):
        changed[ref].append("kernel_start")
    only = {r: v for r, v in changed.items() if r not in relied}
    mentioned = {re.sub(r"\[.*", "", r) for r in list(relied) + list(changed)}
    unused = [
        f"{u}.{f}"
        for u, fs in c["state"].items()
        for f in fs
        if f"{u}.{f}" not in mentioned
    ]
    return {
        "relied": dict(sorted(relied.items())),
        "changed": dict(sorted(only.items())),
        "unused": unused,
    }


def write_trace_md(c, md_path, rows, heads, only_op, use=None):
    """The requirements as a document to read and review, one section per entry point."""
    title = (
        f"LLK contract: {only_op.replace(',', ', ')}"
        if only_op
        else "LLK contract: requirements and where the code meets them"
    )
    src = c["contract"].get("source", "")
    with open(md_path, "w") as fh:
        fh.write(
            f"# {title}\n\nGenerated by `contract_tool.py trace` from `llk_contract.yaml` ({src}). "
        )
        fh.write(
            "Each entry point: the LLK functions that implement it and the compute API functions that call "
            "it; what it requires when called; what it ensures when it returns, with the code lines that "
            "set it; and what it may leave with any value.\n"
        )
        for k, (name, ename, en, start) in enumerate(heads):
            end = heads[k + 1][3] if k + 1 < len(heads) else len(rows)
            op = c["operations"][name]
            fh.write(f"\n## {name}.{ename}\n\n")
            if op.get("description") and (k == 0 or heads[k - 1][0] != name):
                fh.write(" ".join(str(op["description"]).split()) + "\n\n")
            if op.get("operands") and (k == 0 or heads[k - 1][0] != name):
                fh.write(
                    "Operands: "
                    + ", ".join(
                        f"`{o}` on {d['unpacker']}" if d.get("unpacker") else f"`{o}`"
                        for o, d in op["operands"].items()
                    )
                    + "\n\n"
                )
            fh.write("| Thread | LLK function |\n| --- | --- |\n")
            for t, fn in en.get("llk", {}).items():
                tpl = en.get("template", {}).get(t)
                fh.write(
                    f"| {t} | `{fn}`"
                    + (
                        f" with {', '.join(f'{a}={v}' for a, v in tpl.items())}"
                        if tpl
                        else ""
                    )
                    + " |\n"
                )
            if en.get("compute_api"):
                fh.write(
                    "\nCalled by: "
                    + ", ".join(f"`{a}`" for a in en["compute_api"])
                    + "\n"
                )
            for part, label in (("requires", "Requires"), ("ensures", "Ensures")):
                part_rows = [r for r in rows[start:end] if f".{part}:" in r[0]]
                fh.write(
                    f"\n**{label}**" + (" (nothing)\n" if not part_rows else "\n\n")
                )
                if not part_rows:
                    continue
                if part == "requires":
                    fh.write("| Field | Value | Why |\n| --- | --- | --- |\n")
                    for i, v, w, f, where in part_rows:
                        fh.write(f"| `{i.split(':', 1)[1]}` | `{v}` | {w} |\n")
                else:
                    fh.write(
                        "| Field | Value | Set at | Why |\n| --- | --- | --- | --- |\n"
                    )
                    for i, v, w, f, where in part_rows:
                        code = "<br>".join(where[:4]) + (
                            " ..." if len(where) > 4 else ""
                        )
                        fh.write(f"| `{i.split(':', 1)[1]}` | `{v}` | {code} | {w} |\n")
            if en.get("may_change"):
                fh.write(
                    "\n**May change**: "
                    + ", ".join(f"`{x}`" for x in en["may_change"])
                    + "\n"
                )
        if use:
            fh.write(
                "\n## State the entry points rely on\n\n"
                "Derived from the entry points' lists: a field (per unpacker or slot) that some "
                "entry point requires. This is the state a runtime checker such as the Sanitizer "
                "has to track; the rest it can ignore.\n\n| Field | Required by |\n| --- | --- |\n"
            )
            for f, by in use["relied"].items():
                fh.write(f"| `{f}` | {', '.join(by)} |\n")
            fh.write(
                "\n## State no entry point relies on\n\nChanged, or left with any value, "
                "but required by no entry point.\n\n| Field | Changed by |\n| --- | --- |\n"
            )
            for f, by in use["changed"].items():
                fh.write(f"| `{f}` | {', '.join(by)} |\n")
            if use["unused"]:
                fh.write(
                    "\nIn `state` but mentioned by no entry point yet: "
                    + ", ".join(f"`{x}`" for x in use["unused"])
                    + "\n"
                )


# ---------------------------------------------------------------------------------------- check-sequence
def check_sequence(c, seq_path):
    """Walk a call sequence. With `operands` given, values are evaluated for those operands
    (contract_values.py); a value that cannot be evaluated is compared as text, and if the texts
    differ the requirement is reported as undecided, not as broken."""
    sys.path.insert(0, str(HERE))
    from contract_values import UNCHANGED, Params, Unknown, evaluate, fmt_value

    seq = yaml.safe_load(open(seq_path))
    kernel = seq.get("kernel", {})
    operands = seq.get("operands", {})
    state, setter = {}, {}
    ANY = object()

    def value(text, params):
        """(value or None, text, reason); None when it cannot be evaluated."""
        env = {**operands, "op": Params(params), "kernel": kernel}
        try:
            return evaluate(text, env), str(text), ""
        except (Unknown, SyntaxError, KeyError, TypeError) as e:
            return None, str(text), str(e)

    for ref, val in c.get("kernel_start", {}).get("assumes", {}).items():
        state[ref], setter[ref] = value(val, {}), "kernel_start"

    bad = undecided = 0
    print(
        f"check-sequence: {seq_path}"
        + ("" if operands else "  (no operands given: values compared as text)")
    )
    for n, call in enumerate(seq["calls"], 1):
        if isinstance(call, dict):
            ((name, params),) = call.items()
            params = params or {}
        else:
            name, params = call, {}
        op_name, entry = name.split(".")
        op = c["operations"][op_name]
        en = op[entry]
        msgs = []
        for oref, oval in en.get("requires", {}).items():
            ref = canon_ref(op, oref)
            want, wtext, wwhy = value(canon_val(op, oval), params)
            if want is UNCHANGED:
                continue  # 'X if C' with C false: nothing required
            iid = rid(op_name, entry, "requires", oref)
            have = state.get(ref)
            if have is None:
                msgs.append(
                    (
                        "not met",
                        f"{iid} = {fmt_value(want) if want is not None else wtext}: nothing has set it",
                    )
                )
            elif have is ANY:
                msgs.append(
                    (
                        "not met",
                        f"{iid} = {fmt_value(want) if want is not None else wtext}: left unknown by {setter[ref]}",
                    )
                )
            else:
                hv, htext, hwhy = have
                if want is not None and hv is not None:
                    if want != hv:
                        msgs.append(
                            (
                                "not met",
                                f"{iid} = {fmt_value(want)}: is {fmt_value(hv)}, set by {setter[ref]}",
                            )
                        )
                elif wtext != htext:
                    msgs.append(
                        (
                            "undecided",
                            f"{iid} = {wtext}: is {htext}, set by {setter[ref]}; cannot evaluate ({wwhy or hwhy})",
                        )
                    )
            why = en.get("why", {}).get(oref)
            if why and msgs and iid in msgs[-1][1] and msgs[-1][0] == "not met":
                msgs[-1] = (msgs[-1][0], msgs[-1][1] + f"\n          why: {why}")
        nb = sum(1 for k, m in msgs if k == "not met")
        nu = len(msgs) - nb
        status = (
            "ok"
            if not msgs
            else ", ".join(
                x
                for x in (
                    f"{nb} not met" if nb else "",
                    f"{nu} undecided" if nu else "",
                )
                if x
            )
        )
        print(f"  {n:>2}. {name:<24} {status}")
        for k, m in msgs:
            print(f"        - {k}: {m}")
        bad += nb
        undecided += nu
        for oref, oval in en.get("ensures", {}).items():
            ref = canon_ref(op, oref)
            v = value(canon_val(op, oval), params)
            if v[0] is UNCHANGED:
                continue
            state[ref], setter[ref] = (
                v,
                f"call {n} ({rid(op_name, entry, 'ensures', oref)})",
            )
        for oref in en.get("may_change", []):
            ref = canon_ref(op, oref)
            state[ref], setter[ref] = ANY, f"call {n} ({name}, may_change)"
    print(f"check-sequence: {bad} requirements not met, {undecided} undecided")
    return bad


def check_api(c, api_map_path):
    m = json.load(open(api_map_path))
    reach = {
        f"{e['group']}::{e['api']}": {
            t.lower(): set(v["llk"]) for t, v in e["threads"].items()
        }
        for e in m["apis"]
    }
    total = ok = 0
    for name, ename, op, en in entries(c):
        for api in en.get("compute_api", []):
            for t, fn in en.get("llk", {}).items():
                total += 1
                got = reach.get(api, {}).get(t)
                if got is None:
                    print(
                        f"  MISSING  {name}.{ename}: {api} not in the compute API map, or makes no {t} call"
                    )
                elif fn in got:
                    ok += 1
                else:
                    print(
                        f"  MISSING  {name}.{ename}: {api} does not reach {fn} on {t}"
                    )
    print(
        f"check-api: {ok} of {total} (compute API function, thread) pairs reach the LLK function the contract names"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=["validate", "check-code", "check-api", "trace", "check-sequence"],
    )
    ap.add_argument("contract")
    ap.add_argument("target", nargs="?")
    ap.add_argument("--md", help="trace: also write the requirements as Markdown")
    ap.add_argument("--op", help="trace: only these operations, comma-separated")
    a = ap.parse_args()
    c = load(a.contract)
    if a.cmd == "validate":
        sys.exit(1 if validate(c) else 0)
    if a.cmd == "check-code":
        check_code(c, Path(a.target).expanduser().resolve())
    if a.cmd == "check-api":
        check_api(c, a.target)
    if a.cmd == "trace":
        trace(c, Path(a.target).expanduser().resolve(), a.md, a.op)
    if a.cmd == "check-sequence":
        sys.exit(1 if check_sequence(c, a.target) else 0)


if __name__ == "__main__":
    main()
