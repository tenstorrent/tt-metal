"""rc_preflight requirement (main, from rc1b on): every RC flag's effective value, host side == kernel side == recipe.

Host side  : the engine env the build used (engine_env(): ENGINE_ENV + QWEN36_R16_MK_ENV overrides), which build_program
             applies to os.environ around setup_bench / the lazy per-device builds -> what program.py / e1_shim.py read.
Kernel side: the defines each compiled kernel group actually got (KernelDescriptor.defines of device 0's program):
             STREAM / ext (mka2_reader / writer / compute) for every MK_DEFINES flag; the GDN-role kernels (g2_*) for the
             E1_* flags that e1_shim turns into GDN defines (E1_TO_GDN below; the rest are host-only decisions).
Recipe     : RELEASE_RECIPE.md, section "RC engine env": the "MK_DEFINES:" and "Engine env:" paragraphs (KEY=VALUE tokens)
             and the "Explicitly OFF:" list (must be absent or 0 on both sides).

QWEN36_R16_MK_RCFLAGS = print (default: one summary line, per-flag lines on mismatch) | assert (raise on any mismatch;
rc_preflight / acceptance runs) | off.
"""
import os
import re

RECIPE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RELEASE_RECIPE.md")
# E1_* env -> the GDN-role kernel define e1_shim derives from it (value as compiled); others are host-only decisions
E1_TO_GDN = {
    "E1_LZVC": "GDN_LZVC", "E1_WPRE": "GSTEP_WPRE", "E1_ATTHOLD": "GSTEP_ATTHOLD", "E1_WYPAIR": "GSTEP_WYPAIR",
    "E1_WYORD": "GSTEP_WYORD", "E1_CUTST": "GSTEP_CUTST", "E1_KTLATE": "GSTEP_KTLATE", "E1_KTSLATE": "GSTEP_KTSLATE",
    "E1_STNARROW": "GDN_STNARROW", "E1_KTDD": "GDN_KTDD",   # rc1b (7u REPLAY): RC B=8 / VB>1 satisfy their gates
    "E1_GTEXT2": "GDN_GTEXT2", "E1_GBC_WBAR": "GB_WBAR", "E1_GBC_GEARLY": "GB_GEARLY",   # rc1c (7w GTEXT2 / GBC)
    # (E1_GO_SCVC / E1_GO_KSTREAM / E1_GO_SCPASS change generated GDN-role source text, no define: host-only here)
}
# recipe flags that are code defaults (the recipe names them with "default"): an unset host env is the RC value
CODE_DEFAULT = {"E1_EARLY_REPLAY"}
_TOK = re.compile(r"\b([A-Z][A-Z0-9_]{2,})=([0-9A-Za-z._]+)")


def parse_recipe(path=RECIPE):
    txt = open(path).read()
    i = txt.index("## RC engine env")
    j = txt.find("\n## ", i + 1)
    sec = txt[i:j if j > 0 else None]

    def para(head):
        k = sec.find(head)
        if k < 0:
            return ""
        e = sec.find("\n\n", k)
        return sec[k:e if e > 0 else None]

    def toks(t):
        return {k: v.rstrip(".") for k, v in _TOK.findall(t) if k not in ("RED",)}

    eng = toks(para("Engine env:"))
    # MK_DEFINES prose names prerequisites that live in the engine env ("needs MK_AR_RSTD=1 + RED=gdn"): not defines
    md = {k: v for k, v in toks(para("MK_DEFINES:")).items() if k not in eng}
    off = re.findall(r"\b([A-Z][A-Z0-9_]{3,})\b", para("Explicitly OFF:").replace("Explicitly OFF", ""))
    # prose tokens that are not flags ("needs MK_AR_RSTD=1 + RED=gdn ...") are kept: they are real flags with the RC value
    return dict(md=md, eng=eng, off=sorted(set(off) - {"GAR", "RC", "RQBAL", "QOVXO"}))


def _kernel_defines(desc):
    """{kernel basename: {define: value}} of one ProgramDescriptor (union over its kernels of that source)."""
    out = {}
    for k in getattr(desc, "kernels", []) or []:
        name = os.path.basename(str(getattr(k, "kernel_source", "")))
        d = out.setdefault(name, {})
        for kv in list(getattr(k, "defines", []) or []):
            key, val = (kv[0], kv[1]) if isinstance(kv, (tuple, list)) else (str(kv), "")
            d[str(key)] = str(val)
    return out


def check(env, desc0, recipe=None):
    """-> (n_checked, [failure strings])"""
    rec = recipe or parse_recipe()
    host_md = dict(x.split("=", 1) if "=" in x else (x, "1") for x in env.get("MK_DEFINES", "").split(",") if x)
    kd = _kernel_defines(desc0)
    mka = {n: d for n, d in kd.items() if n.startswith("mka2_")}
    gdn = {n: d for n, d in kd.items() if n.startswith("g2_") or "gdn" in n}
    fails, n = [], 0
    for k, v in sorted(rec["md"].items()):
        n += 1
        if host_md.get(k) != v:
            fails.append(f"{k}: recipe {v} host MK_DEFINES {host_md.get(k)}")
        bad = [f"{nm}={d.get(k)}" for nm, d in mka.items() if d.get(k) != v]
        if mka and bad:
            fails.append(f"{k}: recipe {v} kernel {', '.join(bad[:4])}")
    for k, v in sorted(rec["eng"].items()):
        n += 1
        hv = env.get(k, os.environ.get(k))
        if hv != v and not (hv is None and k in CODE_DEFAULT):
            fails.append(f"{k}: recipe {v} host {hv}")
        if k in E1_TO_GDN and gdn:
            g = E1_TO_GDN[k]
            vals = {nm: d.get(g) for nm, d in gdn.items()}
            if not any(x is not None for x in vals.values()) and v not in ("0", ""):
                fails.append(f"{k}: recipe {v} -> kernel {g} absent in every GDN-role kernel ({sorted(vals)[:4]})")
    for k in rec["off"]:
        n += 1
        hv = host_md.get(k, env.get(k))
        kv = [nm for nm, d in kd.items() if d.get(k) not in (None, "0")]
        if hv not in (None, "0") or kv:
            fails.append(f"{k}: recipe OFF, host {hv} kernels {kv[:4]}")
    return n, fails


def report(env, desc0, log=print):
    mode = os.environ.get("QWEN36_R16_MK_RCFLAGS", "print")
    if mode == "off":
        return
    n, fails = check(env, desc0)
    if not fails:
        log(f"[rcflags] {n} flags host==kernel==recipe PASS")
        return
    log(f"[rcflags] {n} flags: {len(fails)} MISMATCH (host / kernel / recipe)")
    for f in fails:
        log(f"[rcflags]   FAIL {f}")
    if mode == "assert":
        raise AssertionError(f"[rcflags] {len(fails)} RC flag mismatches: {fails[:6]}")
