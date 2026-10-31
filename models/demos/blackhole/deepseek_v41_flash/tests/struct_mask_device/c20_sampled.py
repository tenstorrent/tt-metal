# sampled structured rows: membership every round + distribution against softmax over the allowed set (host reference from trace A's logits). needs c00_lib.py
def advance(st, S, allowed, sp, track=None, T=1.0):
    tokens, start, nv = make_block(st, structured=S, zero_fill=True)
    out, dt = verify(tokens, start, nv, mask_block(allowed), sp)
    n = out.shape[1]
    if track is not None:
        r = runner_for_round(int(nv.max()))
        L = logits_rows(r, S)
    committed = torch.full((W, 1 + K), -1, dtype=torch.int32)
    cp = torch.full((W, 1 + K), -1, dtype=torch.int64)
    cnt = torch.ones(W, dtype=torch.int32)
    for s in st["slots"]:
        mm = 0
        if int(nv[s]) > 0:
            while mm < n - 1 and int(tokens[s, 1 + mm]) == int(out[s, mm]):
                mm += 1
        new = [int(out[s, j]) for j in range(mm + 1)]
        committed[s, : mm + 1] = torch.tensor(new, dtype=torch.int32)
        cp[s, : mm + 1] = torch.arange(st["pos"][s], st["pos"][s] + mm + 1)
        cnt[s] = mm + 1
        st["pos"][s] += mm + 1
        st["last"][s] = new[-1]
        if track is not None and s in S:
            idx = torch.tensor(allowed_ids[s])
            p = torch.softmax(L[s][0][idx] / T, dim=0)
            track[s]["exp"] += p
            track[s]["obs"][allowed_ids[s].index(new[0])] += 1 if new[0] in allowed_ids[s] else 0
            track[s]["bad"] += 0 if new[0] in allowed_ids[s] else 1
    d = model.propose_draft_tokens(K, committed, cp, cnt)
    st["drafts"], st["nv"] = d.draft_token_ids.clone(), d.num_valid.clone()
    return dt


gen = torch.Generator().manual_seed(5)
S = [s for s in (1, 5, 9, 14) if s < W] or [0]
allowed_ids = {s: sorted(set(torch.randint(0, V, (6,), generator=gen).tolist())) for s in S}
allowed = {}
for s in S:
    al = torch.zeros(V, dtype=torch.bool)
    al[allowed_ids[s]] = True
    allowed[s] = al
for name, spx, T in (("sampled_t1", sp_sampled(1.0), 1.0), ("sampled_t0.7_p0.9", sp_sampled(0.7, 0, 0.9), 0.7)):
    st = fresh(list(range(W)))
    track = {s: dict(exp=torch.zeros(len(allowed_ids[s])), obs=torch.zeros(len(allowed_ids[s])), bad=0) for s in S}
    R = int(os.environ.get("C20_ROUNDS", "40"))
    t0 = time.time()
    for i in range(R):
        advance(st, S, allowed, spx, track=track, T=T)
    bad = sum(track[s]["bad"] for s in S)
    if name == "sampled_t1":
        zs = []
        for s in S:
            e, o = track[s]["exp"], track[s]["obs"]
            zs += [
                float((o[i] - e[i]) / math.sqrt(max(float(e[i]) * (1 - float(e[i]) / R), 1e-9))) for i in range(len(e))
            ]
        log(
            f"RESULT {name} rounds={R} out_of_set={bad} max|z|={max(abs(z) for z in zs):.2f} exp={[ [round(float(x),1) for x in track[s]['exp']] for s in S]} obs={[ [int(x) for x in track[s]['obs']] for s in S]} secs={time.time()-t0:.1f}"
        )
    else:
        log(f"RESULT {name} rounds={R} out_of_set={bad} (law not compared: top-p) secs={time.time()-t0:.1f}")
log("C20 DONE")
