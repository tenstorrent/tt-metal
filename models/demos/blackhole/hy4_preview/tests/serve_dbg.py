import json
import os
import types

import torch

from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt.generator_vllm import HYV4ForCausalLM

os.environ.setdefault("HY4_SERVE_LAYERS", "2")
res = lambda **kw: print("RES " + json.dumps(kw, default=str), flush=True)
sp = lambda T, k, p, seed=None, n=4: types.SimpleNamespace(
    temperature=[T] * n, top_k=[k] * n, top_p=[p] * n, seed=[seed] * n, enable_log_probs=[False] * n
)
md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "2e9"))))
try:
    m = HYV4ForCausalLM.initialize_vllm_model(None, md, 4, 2048)
    m.warm_serving()
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(1000, 50000, (n,), generator=g).tolist() for n in (37, 90, 600, 61)]
    lens = [len(p) for p in prompts]
    toks = torch.zeros(4, max(lens), dtype=torch.long)
    for i, p in enumerate(prompts):
        toks[i, : lens[i]] = torch.tensor(p)
    first = m.prefill_forward(
        toks, start_pos=[0] * 4, prompt_lens=lens, empty_slots=[0, 1, 2, 3], sampling_params=sp(0.0, 1, 1.0)
    )
    cur = first.reshape(4, 1).long()
    pos = list(lens)
    for st in range(3):
        f = st == 0
        o = m.decode_forward(
            cur,
            torch.tensor(pos),
            sampling_params=sp(0.0, 1, 1.0),
            read_from_device=True,
            reload_inputs=f,
            reload_sampling_params=f,
            reset_sampling_state=f,
        )
        cur = o
        pos = [x + 1 for x in pos]
    res(event="greedy_steps", last=o.reshape(-1).tolist())
    for T, k, p in ((0.9, -1, 1.0), (0.9, 0, 1.0), (0.0, 1, 1.0), (1.0, -1, 1.0)):
        first = m.prefill_forward(
            toks, start_pos=[0] * 4, prompt_lens=lens, empty_slots=[0, 1, 2, 3], sampling_params=sp(T, k, p, 5)
        )
        res(event="first", T=T, k=k, p=p, tokens=first.tolist(), variant=m.dl.variant)
    # direct decode sampled
    cur = first.reshape(4, 1).long()
    o = m.decode_forward(
        cur,
        torch.tensor(lens),
        sampling_params=sp(0.9, -1, 1.0, 5),
        read_from_device=True,
        reload_inputs=True,
        reload_sampling_params=True,
        reset_sampling_state=True,
    )
    res(event="decode0", tokens=o.reshape(-1).tolist(), variant=m._variant, seeds=m._seeds)
finally:
    close_mesh(md)
