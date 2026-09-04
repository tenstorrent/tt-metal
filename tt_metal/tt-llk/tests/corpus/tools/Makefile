# prove_all — one command reproduces the SFPU proof-coverage census.
# (laneMH; corpus/tools).  See PROVE_ALL_README.md.
#
# OUT overrides the run directory; JOBS the bitexact sim parallelism.
OUT  ?= $(HOME)/sfpi-uplift/laneMH-evidence-20260903/run
JOBS ?= 8

.PHONY: prove-all prove-all-selftest prove-all-manifest prove-all-force

## prove every board op at the current pin and emit the master coverage ledger
prove-all:
	python3 prove_all.py --all --out $(OUT) --jobs $(JOBS)

## re-prove every op from scratch (ignore cached verdicts)
prove-all-force:
	python3 prove_all.py --all --force --out $(OUT) --jobs $(JOBS)

## guard routing + census math (fast unit + live 3-op end-to-end)
prove-all-selftest:
	python3 selftest_prove_all.py

## print the data-driven routing manifest census (no proving)
prove-all-manifest:
	@python3 -c "import prove_all as p; b=p.load_board(); m=p.load_manifest(b); \
from collections import Counter; c=Counter(r['engine'] for r in m.values()); \
print('op set == board:', set(m)==set(b), '(%d ops)'%len(m)); print('engines:', dict(c))"
