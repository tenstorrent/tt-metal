# cost of the mask: replay timings of trace S vs masked trace S, upload of the packed mask, total masked vs unmasked round (wall, same state class). needs c00_lib.py
slots = list(range(W))
st = fresh(slots)
tokens, start, nv = make_block(st)
sp = sp_sampled(1.0)
verify(tokens, start, nv, None, sp)  # binds the runner's traces
r = runner_for_round(int(nv.max()))


def tm(fn, reps=50):
    ttnn.synchronize_device(md)
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(md)
    return 1e3 * (time.perf_counter() - t) / reps


ts = tm(lambda: r.tid_s.replay())
tsm = tm(lambda: r.tid_sm.replay())
T = r.U * r.n
packed = torch.full((md.shape[0] * T, WORDS), -1, dtype=torch.int32)
tup = tm(lambda: r.dec.set_mask(packed), 20)
log(
    f"RESULT timing S={ts:.3f} ms  S_masked={tsm:.3f} ms  (+{tsm-ts:.3f} ms)  set_mask host->device={tup:.3f} ms  rows*T={md.shape[0]*T} n={r.n}"
)
# whole round, unmasked sampled vs masked (wall, includes host work); fresh state each pair
al = rand_allowed(0.01, torch.Generator().manual_seed(3))
a_t, m_t = [], []
for it in range(6):
    st = fresh(slots)
    tokens, start, nv = make_block(st)
    _, d0 = verify(tokens, start, nv, None, sp)
    st = fresh(slots)
    tokens, start, nv = make_block(st, structured=[1] if W > 1 else [0])
    _, d1 = verify(tokens, start, nv, mask_block({1 if W > 1 else 0: al}), sp)
    a_t.append(d0)
    m_t.append(d1)
log(f"RESULT round_wall unmasked_sampled={[round(1e3*x,1) for x in a_t]} masked={[round(1e3*x,1) for x in m_t]} ms")
log("C40 DONE")
