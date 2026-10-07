# Adds LTX_E2E_SEEDS (extra warm gens, one per seed) and a per-gen wall-clock log line to the distilled test.
import sys

p = sys.argv[1]
s = open(p).read()
a = """        pipeline.generate(
            prompt,
            output_path=output_filename,"""
b = """        _t_gen = time.perf_counter()
        pipeline.generate(
            prompt,
            output_path=output_filename,"""
assert s.count(a) == 1
s = s.replace(a, b)
a = """        logger.info(f"Saved video to: {output_filename}")
        print_ltx_timing_table("""
b = """        logger.info(f"Saved video to: {output_filename}")
        logger.info(f"E2E_WALL_S gen={number} seed={seed} wall={time.perf_counter() - _t_gen:.3f}")
        print_ltx_timing_table("""
assert s.count(a) == 1
s = s.replace(a, b)
a = """            for extra in range(int(os.environ.get("LTX_E2E_EXTRA_REPLAYS", "0"))):"""
b = """            # LTX_E2E_SEEDS=1,2,..: one more warm gen per listed seed (gen #2, #3, ...).
            for k, extra_seed in enumerate(s for s in os.environ.get("LTX_E2E_SEEDS", "").split(",") if s):
                logger.info(f"=== traced warm pass (gen #{k + 2}, seed {extra_seed}) ===")
                run(prompt=prompt, number=k + 2, seed=int(extra_seed))
            for extra in range(int(os.environ.get("LTX_E2E_EXTRA_REPLAYS", "0"))):"""
assert s.count(a) == 1
s = s.replace(a, b)
if "\nimport time\n" not in s:
    s = s.replace("\nimport os\n", "\nimport os\nimport time\n", 1)
    assert "\nimport time\n" in s
open(p, "w").write(s)
print("patched")
