# exec'd in the test_b16_server_like namespace (model, tok, Sim, torch, ttnn, log, md, ids): helpers of the grammar-mask device tests
import time
from types import SimpleNamespace

W = int(model.B)
K = model.spec_k
V = 129280
WORDS = 4040
U = model.U
TEXTS = [
    "The capital of France is",
    "Write a short poem about the sea and the wind.",
    "Explain what a hash table is in two sentences.",
    "List three prime numbers greater than 100:",
    "Once upon a time in a small village there lived",
    "def fibonacci(n):\n    ",
    "Translate to German: The weather is nice today.",
    "What is the derivative of x squared?",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
    "Summarize the plot of Romeo and Juliet briefly.",
    "In 1969 humans first landed on the",
    "Name the planets of the solar system in order:",
    "How do I make a good cup of coffee?",
    "The theory of relativity states that",
    "Q: What is 12 times 13?\nA:",
    "Dear Sir or Madam, I am writing to",
]
PROMPTS = {s: tok.encode(TEXTS[s % len(TEXTS)]) for s in range(W)}
sim = Sim(model, W)


def sp_greedy(n=W):
    return sim.sp(n)


def sp_sampled(temp=1.0, top_k=0, top_p=1.0, seed=None, n=W):
    return SimpleNamespace(temperature=[temp] * n, top_k=[top_k] * n, top_p=[top_p] * n, seed=[seed] * n)


def fresh(slots, kk=None):
    """prefill the slots (prompt of each), propose drafts. -> state dict"""
    reqs = {s: PROMPTS[s] for s in slots}
    for s in reqs:
        model.release_request(s)
    first = sim.prefill(reqs)
    pos = {s: len(reqs[s]) for s in reqs}
    committed = torch.full((W, 1 + K), -1, dtype=torch.int32)
    cp = torch.full((W, 1 + K), -1, dtype=torch.int64)
    cnt = torch.ones(W, dtype=torch.int32)
    for s in reqs:
        committed[s, 0], cp[s, 0] = first[s], pos[s]
    d = model.propose_draft_tokens(K, committed, cp, cnt)
    return dict(
        slots=list(slots),
        last={s: first[s] for s in reqs},
        pos=pos,
        drafts=d.draft_token_ids.clone(),
        nv=d.num_valid.clone(),
    )


def make_block(st, structured=(), kv=None, zero_fill=False):
    """plugin block [W, 1+K] for the state; structured rows carry no drafts (nv 0). kv: the width an adaptive offer would have (nv set to min(nv, kv))."""
    tokens = torch.zeros(W, 1 + K, dtype=torch.int32)
    start = torch.full((W, 1 + K), -1, dtype=torch.int64)
    nv = st["nv"].clone()
    for s in st["slots"]:
        tokens[s, 0] = st["last"][s]
        tokens[s, 1:] = st["drafts"][s]
        start[s] = torch.arange(st["pos"][s], st["pos"][s] + 1 + K)
        if s in structured:
            nv[s] = 0
            if zero_fill:
                tokens[s, 1:] = 0
        elif kv is not None:
            nv[s] = min(int(nv[s]), kv)
    return tokens, start, nv


def mask_block(allowed_by_slot, width=1 + K):
    """allowed_by_slot {slot: bool[V]} -> int32 [W, width, 4040] (all ones elsewhere); only column 0 meaningful (M1)."""
    blk = torch.full((W, width, WORDS), -1, dtype=torch.int32)
    for s, al in allowed_by_slot.items():
        blk[s, 0] = pack_bool(al)
    return blk


def pack_bool(al):
    a = torch.zeros(WORDS * 32, dtype=torch.int64)
    a[:V] = al.long()
    a = a.reshape(WORDS, 32)
    out = (a << torch.arange(32, dtype=torch.int64)).sum(-1)
    out = torch.where(out >= 2**31, out - 2**32, out)
    return out.to(torch.int32)


def verify(tokens, start, nv, blk=None, sp=None):
    kw = dict(
        spec_mode="argmax_ids",
        num_valid_drafts=nv.clone(),
        accepted_counts=torch.ones(W, dtype=torch.int32),
        sampling_params=sp,
    )
    if blk is not None:
        kw["grammar_bitmask"] = blk
    t0 = time.time()
    out = model.decode_forward(tokens, start, **kw)
    return out.argmax_ids.clone(), time.time() - t0


def runner_for_round(kv):
    if model.spec_adapt is not None:
        return model.spec.runner_for(kv)[1]
    return model.spec


def logits_rows(runner, slots):
    """host fp32 logits [len(slots), n, V] of the LAST round of `runner` (trace A's buffer of the bucket in use), user-major rows"""
    lg = runner.cur_logits
    devs = ttnn.get_device_tensors(lg)
    rows, cols = md.shape
    n = runner.n
    Ur = runner.U
    full = {}
    for r in range(rows):
        full[r] = torch.cat(
            [ttnn.to_torch(devs[r * cols + c]).float().reshape(-1, V // cols) for c in range(cols)], dim=1
        )  # [T, V]
    out = {}
    for s in slots:
        p = model.slots.phys[s]
        r_, u_ = p // Ur, p % Ur
        out[s] = full[r_][u_ * n : (u_ + 1) * n]
    return out


def rand_allowed(density, gen, must=None):
    al = torch.rand(V, generator=gen) < density
    if must is not None:
        al[must] = True
    if not bool(al.any()):
        al[int(torch.randint(0, V, (1,), generator=gen))] = True
    return al


log("LIB READY K=%d U=%d W=%d" % (K, U, W))
