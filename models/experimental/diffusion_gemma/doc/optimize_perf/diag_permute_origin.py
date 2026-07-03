import ttnn, collections, traceback

cnt = collections.Counter()
shp = {}


def wrap(name, orig):
    def g(t, *a, **k):
        st = traceback.extract_stack(limit=12)
        site = next(
            (
                f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}"
                for f in reversed(st)
                if "diffusion_gemma" in f.filename and "diag_" not in f.filename
            ),
            name + ":?",
        )
        cnt[(name, site)] += 1
        try:
            shp[(name, site)] = tuple(t.shape)
        except:
            pass
        return orig(t, *a, **k)

    return g


ttnn.permute = wrap("permute", ttnn.permute)
ttnn.transpose = wrap("transpose", ttnn.transpose)
import sys

sys.argv = ["x"]
from models.experimental.diffusion_gemma.doc.optimize_perf import prof_denoise_step as P

try:
    P.run(2, 256, 1, "The capital of France is", 512, do_trace=False)
except Exception as e:
    import traceback as tb

    tb.print_exc()
print("\n=== permute/transpose call sites (count, shape, site) ===")
for (nm, site), c in cnt.most_common(20):
    print(f"{nm:9s} {c:4d}  {shp.get((nm,site))}  {site}")
