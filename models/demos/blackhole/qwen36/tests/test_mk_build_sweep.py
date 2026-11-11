"""In-model engine BUILD sweep: which engine flag subsets build (incl. cbplan verify / static-CB clash) inside the real
model process. One model load (flag off, prod env), then build_program for each config of SWEEP (';'-separated list of
"name|MK_DEFINES-removals|env overrides"), reporting ok / the clash numbers. No engine launch.
  QWEN36_R16_MEGAKERNEL unset; pytest models/demos/blackhole/qwen36/tests/test_mk_build_sweep.py -s -x
"""
import json
import os
import re

from loguru import logger

from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import _marks

NL = int(os.environ.get("SWEEP_NL", "8"))
DEFAULT = ";".join([
    "M3||",
    "noKVA||MKA_KVA=0",
    "noKVA_noINPROJ|MK_GDN_INPROJ=1|MKA_KVA=0",
    "noKVA_noINPROJ_noRQBAL|MK_GDN_INPROJ=1,MKA_RQBAL=1|MKA_KVA=0",
    "noKVA_noINPROJ_noRQPOS|MK_GDN_INPROJ=1,MKA_RQBAL=1,MKA_RQPOS=1|MKA_KVA=0",
    "noKVA_noGAR|MK_GDN_INPROJ=1,MK_REAL_GAR=1|MKA_KVA=0",
    "noGDN24|MK_GDN_INPROJ=1,MK_REAL_GAR=1|MKA_KVA=0 MKA_GDN24=0 MKA_GDN_WAIT=0 MKA_GDNOUT=0",
    "noGDN24_KVA|MK_GDN_INPROJ=1,MK_REAL_GAR=1|MKA_GDN24=0 MKA_GDN_WAIT=0 MKA_GDNOUT=0",
])
SWEEP = open(os.environ["SWEEP_FILE"]).read().strip() if os.environ.get("SWEEP_FILE") else os.environ.get("SWEEP", DEFAULT)
OUT = os.environ.get("SWEEP_OUT", "")


@_marks
def test_mk_build_sweep(mesh_device, monkeypatch):
    assert os.environ.get("QWEN36_R16_MEGAKERNEL", "") in ("", "0"), "run flag-off: the sweep builds the engines itself"
    monkeypatch.setattr(tc, "B", 8)
    model, eng, _tok, _p = tc._setup(mesh_device)
    from models.demos.blackhole.qwen36.tt.megakernel.engine import program as EP
    from models.demos.blackhole.qwen36.tt.megakernel.integ import stage1 as S1
    from models.demos.blackhole.qwen36.tt.megakernel.integ import weights_mk as WM

    mesh = model.mesh_device
    lt = ["attention" if l.is_full_attention else "gdn" for l in model.layers][:NL]
    kv = [d for d in S1.kv_descriptors(model) if d["layer"] < NL]
    gdn = [d for d in S1.gdn_descriptors(model) if d["layer"] < NL]
    w = WM.repack(model, layouts=EP.weight_layouts(mesh), only_layers=list(range(NL)))
    base_defs = EP.ENGINE_ENV["MK_DEFINES"].split(",")
    res = []
    for item in [s for s in SWEEP.split(";") if s.strip()]:
        name, drop, envs = (item.split("|") + ["", ""])[:3]
        defs = [d for d in base_defs if d not in set(x for x in drop.split(",") if x)]
        mk_env = " ".join([f"MK_DEFINES={','.join(defs)}"] + [e for e in envs.split() if e])
        monkeypatch.setenv("QWEN36_R16_MK_ENV", mk_env)
        monkeypatch.setenv("QWEN36_R16_MK_NL", str(NL))
        r = dict(name=name, drop=drop, env=envs)
        try:
            p = EP.build_program(mesh, 8, 4, lt, w, kv, gdn)
            r.update(ok=True, l1pool=p.info.get("l1pool"))
            del p
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            m = re.search(r"L1 buffer allocated at (\d+) and static circular buffer region ends at (\d+)", msg)
            r.update(ok=False, err=msg.splitlines()[0][:300] if not m else "clash",
                     l1_floor=int(m.group(1)) if m else None, cb_end=int(m.group(2)) if m else None,
                     deficit=(int(m.group(2)) - int(m.group(1))) if m else None)
            m2 = re.search(r"core range \[([^\]]+)\]", msg)
            if m2:
                r["cores"] = m2.group(1)
        logger.info(f"[sweep] {json.dumps(r)}")
        res.append(r)
    if OUT:
        with open(OUT, "a") as f:
            f.write(json.dumps(res) + "\n")
