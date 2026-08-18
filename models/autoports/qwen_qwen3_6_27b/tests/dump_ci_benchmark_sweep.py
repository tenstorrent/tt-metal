import sys

sys.path.insert(0, ".")
from reference_config.benchmarking.benchmark_config import get_benchmark_config  # noqa
from workflows.model_spec import MODEL_SPECS  # noqa

spec = None
for _, s in MODEL_SPECS.items():
    if getattr(s, "model_name", None) == "Qwen3.6-27B":
        spec = s
        break
if spec is None:
    print("model spec not found")
    raise SystemExit(1)

cfg = get_benchmark_config(spec)
rows = []
for task in cfg.tasks:
    for dev, params in task.param_map.items():
        dname = getattr(dev, "name", str(dev))
        for p in params:
            if p.isl is None or p.osl is None or p.task_type != "text":
                continue
            rows.append((dname, p.isl, p.osl, p.max_concurrency, p.num_prompts,
                         bool(p.targets)))

print("  %-10s %6s %6s %10s %8s %8s" % ("device", "isl", "osl", "max_conc", "n", "targets"))
for r in sorted(set(rows)):
    print("  %-10s %6s %6s %10s %8s %8s" % r)

print()
p300 = [r for r in set(rows) if "300" in r[0].lower()]
long_hi = [r for r in p300 if r[3] and r[3] >= 8 and r[2] and r[2] >= 1024]
print("  p300x2 points total:", len(p300))
print("  p300x2 points with max_conc>=8 AND osl>=1024:", len(long_hi))
for r in sorted(long_hi):
    print("    ", r)
