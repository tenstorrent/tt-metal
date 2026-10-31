# law of the masked sampler vs the unmasked top_k=6 sampler (same temperature 1, same state): per row allowed = the row's top-6 logits (read from trace A of a SAMPLED round: a greedy round runs the
# single trace whose logits are not kept). Identical prompts in every row (more draws per state). needs c00_lib.py
saved = dict(PROMPTS)
for s in range(W):
    PROMPTS[s] = saved[0]
slots = list(range(W))
PAIRS = int(os.environ.get("C22_PAIRS", "12"))
res = {"masked": torch.zeros(6), "topk6": torch.zeros(6), "exp": torch.zeros(6)}
oos = {"masked": 0, "topk6": 0}
dd, drow = [], []
for it in range(PAIRS):
    st = fresh(slots)
    tokens, start, nv = make_block(st)
    out_u, _ = verify(tokens, start, nv, None, sp_sampled(1.0, 6, 1.0))
    r = runner_for_round(int(nv.max()))
    L0 = logits_rows(r, slots)
    tops = {s_: L0[s_][0].topk(6).indices.tolist() for s_ in slots}
    for s_ in slots:
        res["exp"] += torch.softmax(L0[s_][0][tops[s_]], dim=0)
    drow.append(max(float((L0[s_][0] - L0[0][0]).abs().max()) for s_ in slots))
    for s_ in slots:
        t = int(out_u[s_, 0])
        if t in tops[s_]:
            res["topk6"][tops[s_].index(t)] += 1
        else:
            oos["topk6"] += 1
    als = {}
    for s_ in slots:
        al = torch.zeros(V, dtype=torch.bool)
        al[tops[s_]] = True
        als[s_] = al
    st = fresh(slots)
    tokens, start, nv = make_block(st, structured=slots)
    out_m, _ = verify(tokens, start, nv, mask_block(als), sp_sampled(1.0))
    L1 = logits_rows(r, slots)
    dd.append(max(float((L1[s_][0] - L0[s_][0]).abs().max()) for s_ in slots))
    for s_ in slots:
        t = int(out_m[s_, 0])
        if t in tops[s_]:
            res["masked"][tops[s_].index(t)] += 1
        else:
            oos["masked"] += 1
log(
    f"RESULT law_control pairs={PAIRS} exp={[round(float(x),1) for x in res['exp']]} masked={[int(x) for x in res['masked']]} topk6_unmasked={[int(x) for x in res['topk6']]} oos={oos} max|L1-L0| across fresh states={max(dd):.3g} row-vs-row0 (same prompt)={max(drow):.3g}"
)
PROMPTS.update(saved)
log("C22 DONE")
