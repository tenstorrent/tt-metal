# M2 end to end (model side): with a state-independent ban list B (the same mask at every position) the SPECULATIVE masked decode (drafts, per-position masks, accept / commit on device)
# must produce exactly the token stream of the plain masked decode (one token per round, nv = 0). Needs DSV41_STRUCTURED_M2=1 and c00_lib.py.
N = int(os.environ.get("C31_TOKENS", "48"))


def walk(st, tokens, nv, out, com):
    n = out.shape[1]
    committed = torch.full((W, 1 + K), -1, dtype=torch.int32)
    cp = torch.full((W, 1 + K), -1, dtype=torch.int64)
    cnt = torch.ones(W, dtype=torch.int32)
    acc = 0
    for s in st["slots"]:
        mm = 0
        if int(nv[s]) > 0:
            while mm < n - 1 and int(tokens[s, 1 + mm]) == int(out[s, mm]):
                mm += 1
        new = [int(out[s, j]) for j in range(mm + 1)]
        com[s].extend(new)
        acc += mm
        committed[s, : mm + 1] = torch.tensor(new, dtype=torch.int32)
        cp[s, : mm + 1] = torch.arange(st["pos"][s], st["pos"][s] + mm + 1)
        cnt[s] = mm + 1
        st["pos"][s] += mm + 1
        st["last"][s] = new[-1]
    d = model.propose_draft_tokens(K, committed, cp, cnt)
    st["drafts"], st["nv"] = d.draft_token_ids.clone(), d.num_valid.clone()
    return acc


def decode(slots, banned, mode, steps=None):
    st = fresh(slots)
    com = {s: [st["last"][s]] for s in slots}
    allowed = torch.ones(V, dtype=torch.bool)
    if banned:
        allowed[banned] = False
    acc = rounds = 0
    while min(len(com[s]) for s in slots) < N:
        if mode == "plain":
            tokens, start, nv = make_block(st, structured=slots)
            blk = mask_block({s: allowed for s in slots})
        elif mode == "unmasked":
            tokens, start, nv = make_block(st)
            blk = None
        else:
            tokens, start, nv = make_block(st)
            blk = mask_block_pos({s: {j: allowed for j in range(1 + K)} for s in slots})
        out, _ = verify(tokens, start, nv, blk, sp_greedy())
        acc += walk(st, tokens, nv, out, com)
        rounds += 1
    return com, acc, rounds


def mask_block_pos(d, width=1 + K):
    blk = torch.full((W, width, WORDS), -1, dtype=torch.int32)
    for s, cols in d.items():
        for j, al in cols.items():
            blk[s, j] = pack_bool(al)
    return blk


slots = list(range(W))
com_u, acc_u, r_u = decode(slots, [], "unmasked")
freq = {}
for s in slots:
    for t in com_u[s][1:N]:
        freq[t] = freq.get(t, 0) + 1
banned = [t for t, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:6]]
com_p, acc_p, r_p = decode(slots, banned, "plain")
if os.environ.get("DSV41_STRUCTURED_M2") == "1":
    com_s, acc_s, r_s = decode(slots, banned, "spec")
    same = {s: com_p[s][:N] == com_s[s][:N] for s in slots}
    hit_banned = sum(1 for s in slots for t in com_s[s][1:N] if t in banned)
    log(
        f"RESULT m2_stream tokens={N} banned={banned} plain_rounds={r_p} spec_rounds={r_s} spec_accepted_drafts={acc_s} unmasked_accepted={acc_u} "
        f"banned_tokens_in_spec_stream={hit_banned} identical_rows={sum(same.values())}/{W} differs_from_unmasked={sum(1 for s in slots if com_p[s][:N] != com_u[s][:N])}/{W}"
    )
    for s in slots:
        if not same[s]:
            i = next(i for i in range(N) if com_p[s][i] != com_s[s][i])
            log(f"  MISMATCH slot {s} first diff at {i}: plain {com_p[s][i-2:i+3]} spec {com_s[s][i-2:i+3]}")
else:
    log(f"RESULT m2_stream skipped (no M2); plain masked rounds={r_p}")
log("C31 DONE")
