import itertools, sys, time, pytest
from models.tt_dit.tests.models.minimax_h3 import test_sla_attention_minimax_h3 as a, test_sla_selection_minimax_h3 as s

fails = 0
for case, sp, qbt in itertools.product(sorted(a._SEQ), [1, 8], [1, 2]):
    mp = pytest.MonkeyPatch()
    t0 = time.time()
    try:
        a.test_sla_attention_matches_lightx2v(case, sp, qbt, mp)
        r = "PASS"
    except Exception as e:
        r = f"FAIL {e!r}"[:300]
        fails += 1
    finally:
        mp.undo()
    print(case, sp, qbt, r, f"{time.time()-t0:.0f}s", flush=True)
for case, sp, sparsity, qbt in itertools.product(sorted(s._CASES), [1, 32], [0.85, 0.9], [1, 2]):
    s.test_sla_selection_matches_lightx2v(case, sp, sparsity, qbt)
print("selection tests pass")
sys.exit(fails)
