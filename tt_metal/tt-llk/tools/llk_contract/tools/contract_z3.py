#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Proofs over the LLK contract with Z3, for every format and tile shape at once.

check-sequence answers "is this sequence correct for these operands?". This answers "is it correct for
all operands?": every format of every operand (all Blackhole DataFormat values), every tile shape
(face_r_dim in 1, 2, 4, 8, 16; num_faces in 1, 2, 4), every value of every operation parameter and
kernel setting. Z3 either proves a requirement holds for all of them, or gives a counterexample:
values for which it does not.

Values are translated as follows:
  - format predicates, sizes, conditions and arithmetic: as the LLK code computes them
    (contract_values.py has the same definitions);
  - every other function (canonical_z_stride, matmul, datacopy ...) is an uninterpreted function:
    Z3 knows only that the same function of equal arguments gives equal values. So a requirement is
    proved only if it holds whatever those hardware functions compute.
  - `X if C` in an ensures: the field becomes X when C holds and keeps its value otherwise;
    in a requires: the field must be X when C holds.

What is proved, per operation (B) and pair of operations (A, B):
  establish   configure, B.init, B.execute: every requirement of B.execute holds
  restore     configure, B.init, B.uninit: every field B.init changes is back to configure's value
  after       configure, A.init, A.execute, A.uninit, B.init, B.execute: B, with its own init after A,
              gets every requirement (A leaves nothing B's init does not set again)
  between     configure, B.init, A.<entry>, B.execute: which requirements of B fail, and when
              (expected to fail for shared program state; listed for the Sanitizer)
and any sequence file, for all operands (prove-sequence).

Usage, from the folder above tools/:
  python3 tools/contract_z3.py prove          contract/llk_contract.yaml
  python3 tools/contract_z3.py prove-sequence contract/llk_contract.yaml examples/sequences/tilize_ok.yaml
"""

import ast
import re
import sys
from pathlib import Path

import yaml
import z3

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from contract_tool import canon_ref, canon_val, load, rid  # noqa: E402
from contract_values import DATUM_BYTES, FORMATS, _split_if  # noqa: E402

FORMAT_NAMES = list(FORMATS)
Fmt, FMT = z3.EnumSort("Format", FORMAT_NAMES)
F = dict(zip(FORMAT_NAMES, FMT))

Val = z3.Datatype("Val")
Val.declare("I", ("i", z3.IntSort()))
Val.declare("Fm", ("f", Fmt))
Val.declare("K", ("k", z3.StringSort()))  # a named constant: pass_through, default
Val = Val.create()

FACE_R = (1, 2, 4, 8, 16)
NUM_FACES = (1, 2, 4)


class Untranslatable(Exception):
    pass


class Sym:
    """Symbolic operands, parameters and kernel settings, with their domain constraints."""

    def __init__(self):
        self.vars, self.domain = {}, []

    def var(self, name, kind):
        if name not in self.vars:
            if kind == "fmt":
                v = z3.Const(name, Fmt)
            elif kind == "bool":
                v = z3.Int(name)
                self.domain.append(z3.Or(v == 0, v == 1))
            else:
                v = z3.Int(name)
                self.domain.append(v >= 0)
            self.vars[name] = v
        return self.vars[name]

    def operand(self, inst):
        """srcA, srcB or out, as a nested dict of Z3 variables."""
        fmts = ("dest", "l1") if inst == "out" else ("l1", "unpacked")
        fr = self.var(f"{inst}.tile.face_r_dim", "int")
        nf = self.var(f"{inst}.tile.num_faces", "int")
        if not any(str(fr) in str(d) and "Or" in str(d) for d in self.domain):
            self.domain.append(z3.Or([fr == x for x in FACE_R]))
            self.domain.append(z3.Or([nf == x for x in NUM_FACES]))
        return {
            "format": {f: self.var(f"{inst}.format.{f}", "fmt") for f in fmts},
            "tile": {
                "face_r_dim": fr,
                "num_faces": nf,
                "partial_face": self.var(f"{inst}.tile.partial_face", "bool"),
            },
        }


class Params(dict):
    def __init__(self, sym, prefix):
        super().__init__()
        self.sym, self.prefix = sym, prefix

    def get_attr(self, attr):
        if attr not in self:
            self[attr] = self.sym.var(f"{self.prefix}.{attr}", "int")
        return self[attr]


def to_val(x):
    if isinstance(x, z3.ExprRef):
        s = x.sort()
        if s == Val:
            return x
        if s == Fmt:
            return Val.Fm(x)
        if s == z3.IntSort():
            return Val.I(x)
        if s == z3.BoolSort():
            return Val.I(z3.If(x, 1, 0))
    if isinstance(x, bool):
        return Val.I(int(x))
    if isinstance(x, int):
        return Val.I(x)
    if isinstance(x, str):
        return Val.Fm(F[x]) if x in F else Val.K(z3.StringVal(x))
    if isinstance(x, tuple):  # a tuple of values: one value, equal when all parts are
        return uf(f"tuple{len(x)}", len(x))(*[to_val(e) for e in x])
    raise Untranslatable(f"no value for {x!r}")


def to_int(x):
    if isinstance(x, z3.ExprRef):
        if x.sort() == z3.IntSort():
            return x
        if x.sort() == z3.BoolSort():
            return z3.If(x, 1, 0)
        if x.sort() == Val:
            return z3.If(Val.is_I(x), Val.i(x), -1)
    if isinstance(x, (int, bool)):
        return z3.IntVal(int(x))
    raise Untranslatable(f"not a number: {x!r}")


def to_bool(x):
    if isinstance(x, z3.ExprRef) and x.sort() == z3.BoolSort():
        return x
    return to_int(x) != 0


def to_fmt(x):
    if isinstance(x, str) and x in F:
        return F[x]
    if isinstance(x, z3.ExprRef) and x.sort() == Fmt:
        return x
    raise Untranslatable(f"not a format: {x!r}")


def flat(d):
    """Leaves of a nested dict of Z3 values, in a fixed order."""
    out = []
    for k in sorted(d):
        v = d[k]
        out += flat(v) if isinstance(v, dict) else [v]
    return out


UF = {}


def uf(name, n):
    key = (name, n)
    if key not in UF:
        UF[key] = z3.Function(name, *([Val] * n), Val) if n else z3.Const(name, Val)
    return UF[key]


def known(name, args):
    if name == "is_8bit":
        f = to_fmt(args[0])
        return z3.Or([f == F[x] for x in ("Int8", "UInt8", "Fp8_e4m3", "Lf8")])
    if name == "is_int8_or_int32":
        f = to_fmt(args[0])
        return z3.Or([f == F[x] for x in ("Int8", "UInt8", "Int32")])
    if name == "is_fp8_e4m3":
        return to_fmt(args[0]) == F["Fp8_e4m3"]
    if name == "tile_datums":
        t = args[0]
        return t["face_r_dim"] * t["num_faces"] * 16
    if name == "datum_size":
        f = to_fmt(args[0])
        e = z3.IntVal(1)
        for k, b in DATUM_BYTES.items():
            e = z3.If(f == F[k], b, e)
        return e
    return None


def tr(n, env):
    if isinstance(n, ast.Constant):
        return int(n.value) if isinstance(n.value, bool) else n.value
    if isinstance(n, ast.Name):
        if n.id in env:
            return env[n.id]
        if n.id in ("True", "False"):
            return int(n.id == "True")
        return n.id
    if isinstance(n, ast.Attribute):
        base = tr(n.value, env)
        if isinstance(base, Params):
            return base.get_attr(n.attr)
        if isinstance(base, dict) and n.attr in base:
            return base[n.attr]
        raise Untranslatable(f"{ast.unparse(n)}")
    if isinstance(n, ast.Tuple):
        return tuple(tr(e, env) for e in n.elts)
    if isinstance(n, ast.Call):
        name = ast.unparse(n.func)
        args = [tr(a, env) for a in n.args]
        k = known(name, args)
        if k is not None:
            return k
        vals = []
        for a in args:
            if isinstance(a, dict):
                vals += [to_val(x) for x in flat(a)]
            else:
                vals.append(to_val(a))
        f = uf(name, len(vals))
        return f(*vals) if vals else f
    if isinstance(n, ast.BinOp):
        a, b = to_int(tr(n.left, env)), to_int(tr(n.right, env))
        if isinstance(n.op, ast.Add):
            return a + b
        if isinstance(n.op, ast.Sub):
            return a - b
        if isinstance(n.op, ast.Mult):
            return a * b
    if isinstance(n, ast.BoolOp):
        vs = [to_bool(tr(v, env)) for v in n.values]
        return z3.And(vs) if isinstance(n.op, ast.And) else z3.Or(vs)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
        return z3.Not(to_bool(tr(n.operand, env)))
    if isinstance(n, ast.IfExp):
        return z3.If(
            to_bool(tr(n.test, env)), to_val(tr(n.body, env)), to_val(tr(n.orelse, env))
        )
    if isinstance(n, ast.Compare) and len(n.ops) == 1:
        a, b = tr(n.left, env), tr(n.comparators[0], env)
        op = n.ops[0]
        if isinstance(op, (ast.Eq, ast.NotEq)):
            r = to_val(a) == to_val(b)
            return r if isinstance(op, ast.Eq) else z3.Not(r)
        if isinstance(op, (ast.In, ast.NotIn)):  # codespell:ignore notin
            r = z3.Or([to_val(a) == to_val(x) for x in b])
            return r if isinstance(op, ast.In) else z3.Not(r)
        a, b = to_int(a), to_int(b)
        return {ast.Gt: a > b, ast.GtE: a >= b, ast.Lt: a < b, ast.LtE: a <= b}[
            type(op)
        ]
    raise Untranslatable(ast.unparse(n))


def translate(text, env):
    """(condition, value): the field is `value` when `condition` holds ('X if C'), else always."""
    s = str(text).strip()
    cond = _split_if(s)
    if cond:
        x, c = cond
        return to_bool(tr(ast.parse(c, mode="eval").body, env)), to_val(
            tr(ast.parse(x, mode="eval").body, env)
        )
    return z3.BoolVal(True), to_val(tr(ast.parse(s, mode="eval").body, env))


class Run:
    """A symbolic walk through a sequence of calls."""

    def __init__(self, c, sym, params_by_op=None):
        self.c, self.sym = c, sym
        self.ops = {i: sym.operand(i) for i in ("srcA", "srcB", "out")}
        self.kernel = Params(sym, "kernel")
        self.params = params_by_op or {}
        self.state, self.setter = {}, {}
        self.unknown = 0
        for ref, val in c.get("kernel_start", {}).get("assumes", {}).items():
            _, v = translate(val, self.env(None))
            self.state[ref], self.setter[ref] = v, "kernel_start"

    def env(self, op_name):
        if op_name not in self.params:
            p = Params(self.sym, f"op[{op_name}]")
            op = self.c["operations"].get(op_name, {})
            # every parameter the operation's values use, so that matmul(op) has one meaning
            text = yaml.safe_dump(
                {
                    e: op.get(e)
                    for e in (
                        "hw_configure",
                        "reconfigure",
                        "init",
                        "execute",
                        "uninit",
                    )
                }
            )
            supported = (op.get("supports") or {}).get("params", {})
            for attr in sorted(set(re.findall(r"\bop\.(\w+)", text))):
                v = p.get_attr(attr)
                if (
                    attr in supported
                ):  # the supported values, where the operation lists them
                    self.sym.domain.append(
                        z3.Or(
                            [
                                v == int(x)
                                for x in supported[attr]
                                if isinstance(x, (int, bool))
                            ]
                        )
                    )
            self.params[op_name] = p
        return {**self.ops, "op": self.params[op_name], "kernel": self.kernel}

    def fresh(self, why):
        self.unknown += 1
        return z3.Const(f"unknown{self.unknown}", Val)

    def call(self, op_name, entry, label):
        """Obligations (id, holds-condition) for the call's requires; then apply its ensures."""
        op = self.c["operations"][op_name]
        en = op[entry]
        env = self.env(op_name)
        obligations = []
        for oref, oval in en.get("requires", {}).items():
            ref = canon_ref(op, oref)
            iid = rid(op_name, entry, "requires", oref)
            cond, want = translate(canon_val(op, oval), env)
            if ref not in self.state:
                obligations.append((iid, None, "nothing sets it", label))
                continue
            obligations.append(
                (
                    iid,
                    z3.Implies(cond, self.state[ref] == want),
                    self.setter[ref],
                    label,
                )
            )
        for oref, oval in en.get("ensures", {}).items():
            ref = canon_ref(op, oref)
            cond, v = translate(canon_val(op, oval), env)
            old = self.state.get(ref)
            self.state[ref] = (
                v
                if old is None or z3.is_true(z3.simplify(cond))
                else z3.If(cond, v, old)
            )
            self.setter[ref] = f"{label} ({rid(op_name, entry, 'ensures', oref)})"
        for oref in en.get("may_change", []):
            ref = canon_ref(op, oref)
            self.state[ref], self.setter[ref] = self.fresh(ref), f"{label} (may_change)"
        return obligations


def check(sym, formula):
    """None if `formula` holds for all values in the domain; otherwise a counterexample."""
    s = z3.Solver()
    s.add(sym.domain)
    s.add(z3.Not(formula))
    r = s.check()
    if r == z3.unsat:
        return None
    if r == z3.unknown:
        return "unknown"
    m = s.model()
    used = {str(d) for d in z3.z3util.get_vars(formula)}
    shown = []
    for name, v in sym.vars.items():
        if name in used:
            val = m.eval(v, model_completion=True)
            shown.append(f"{name}={val}")
    return ", ".join(sorted(shown)) or "(any values)"


def run_sequence(c, calls):
    """[(call label, id, result)] for a list of (op, entry); result None = proved."""
    sym = Sym()
    r = Run(c, sym)
    out = []
    for n, (op_name, entry) in enumerate(calls, 1):
        label = f"call {n} {op_name}.{entry}"
        for iid, f, setter, _ in r.call(op_name, entry, label):
            if f is None:
                out.append((label, iid, f"not met for any operands: {setter}"))
            else:
                ce = check(sym, f)
                if ce is not None:
                    out.append((label, iid, f"fails when {ce}; last set by {setter}"))
    return out, r, sym


def ops_with(c, *entries_):
    return [n for n, op in c["operations"].items() if all(e in op for e in entries_)]


def prove(c):
    total_fail = 0
    print("prove: every format, tile shape, parameter and kernel setting\n")
    cfg = [("configure", "hw_configure")]

    print("establish   configure, B.init, B.execute")
    for b in ops_with(c, "init", "execute"):
        res, _, _ = run_sequence(c, cfg + [(b, "init"), (b, "execute")])
        n = sum(len(c["operations"][b]["execute"].get("requires", {})) for _ in [0])
        print(f"  {b:<10} {'PROVED' if not res else 'FAILS'} ({n} requirements)")
        for _, iid, why in res:
            print(f"      {iid}: {why}")
        total_fail += len(res)

    print("\nrestore     configure, B.init, B.uninit: back to configure's value")
    for b in ops_with(c, "init", "uninit"):
        sym = Sym()
        r = Run(c, sym)
        r.call("configure", "hw_configure", "configure")
        base = dict(r.state)
        r.call(b, "init", "init")
        ensured = {
            canon_ref(c["operations"][b], x)
            for x in c["operations"][b]["init"].get("ensures", {})
        }
        changed = [
            ref
            for ref in r.state
            if ref in base and ref in ensured and not z3.eq(r.state[ref], base[ref])
        ]
        r.call(b, "uninit", "uninit")
        fails = []
        for ref in changed:
            ce = check(sym, r.state[ref] == base[ref])
            if ce is not None:
                fails.append((ref, ce))
        print(
            f"  {b:<10} {'PROVED' if not fails else 'NOT RESTORED'} ({len(changed)} fields changed by init)"
        )
        for ref, ce in fails:
            print(f"      {ref}: differs from configure's value when {ce}")

    print("\nafter       configure, A.init, A.execute, A.uninit, B.init, B.execute")
    ops = ops_with(c, "init", "execute")
    for a in ops:
        for b in ops:
            if a == b:
                continue
            seq = (
                cfg
                + [(a, "init"), (a, "execute")]
                + ([(a, "uninit")] if "uninit" in c["operations"][a] else [])
                + [(b, "init"), (b, "execute")]
            )
            res, _, _ = run_sequence(c, seq)
            res = [x for x in res if x[0].endswith(f"{b}.execute")]
            print(f"  {a} then {b:<10} {'PROVED' if not res else 'FAILS'}")
            for _, iid, why in res:
                print(f"      {iid}: {why}")
            total_fail += len(res)

    print(
        "\nbetween     configure, B.init, A.<entry>, B.execute (expected to fail for shared state)"
    )
    for b in ops:
        for a, opa in c["operations"].items():
            if a in (b, "configure"):
                continue
            for e in ("init", "execute", "uninit"):
                if e not in opa:
                    continue
                res, _, _ = run_sequence(c, cfg + [(b, "init"), (a, e), (b, "execute")])
                res = [x for x in res if x[0].endswith(f"{b}.execute")]
                if res:
                    print(
                        f"  {a}.{e} between {b}.init and {b}.execute: {len(res)} requirement{'' if len(res) == 1 else 's'} fail"
                    )
                    for _, iid, why in res:
                        print(f"      {iid}: {why.split('; last set by')[0]}")
    print(f"\nprove: {total_fail} requirements fail in establish and after")
    return total_fail


def prove_sequence(c, path):
    seq = yaml.safe_load(open(path))
    calls = []
    for call in seq["calls"]:
        name = next(iter(call)) if isinstance(call, dict) else call
        calls.append(tuple(name.split(".")))
    res, _, _ = run_sequence(c, calls)
    print(f"prove-sequence: {path}, for every format, tile shape and parameter")
    last = None
    for label, iid, why in res:
        if label != last:
            print(f"  {label}")
            last = label
        print(f"      {iid}: {why}")
    print(
        f"prove-sequence: {'PROVED' if not res else f'{len(res)} requirements can fail'}"
    )
    return len(res)


def main():
    cmd, contract = sys.argv[1], load(sys.argv[2])
    if cmd == "prove":
        sys.exit(1 if prove(contract) else 0)
    if cmd == "prove-sequence":
        sys.exit(1 if prove_sequence(contract, sys.argv[3]) else 0)
    print(__doc__)


if __name__ == "__main__":
    main()
