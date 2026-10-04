# Current formal campaigns use formal_campaign.py.  prove_all targets below
# reproduce the historical pin-59 coverage census only.
#
# OUT overrides the run directory; JOBS the bitexact sim parallelism.
OUT  ?= $(if $(CRAQ_PROVE_OUT),$(CRAQ_PROVE_OUT),$(HOME)/craq-keep/prove-all)
JOBS ?= 8
FORMAL_OUT ?= $(HOME)/craq-keep/formal-current
FORMAL_OPS ?= *

.PHONY: prove-all prove-all-anchor prove-all-selftest prove-all-manifest \
        prove-all-force shard-selftest formal formal-selftest

## prove current LLK/knob selections; requires FORMAL_SIM and FORMAL_SELECTION
formal:
	@test -n "$(FORMAL_SIM)" || (echo "set FORMAL_SIM" >&2; exit 2)
	@test -n "$(FORMAL_SELECTION)" || (echo "set FORMAL_SELECTION" >&2; exit 2)
	python3 formal_campaign.py --sim "$(FORMAL_SIM)" \
		--selection "$(FORMAL_SELECTION)" --ops "$(FORMAL_OPS)" --out "$(FORMAL_OUT)"

formal-selftest:
	python3 selftest_formal_campaign.py

## prove every board op at the current pin and emit the master coverage ledger
prove-all:
	python3 prove_all.py --all --out $(OUT) --jobs $(JOBS)

## same, but first take the bitexact leg's DEVICE ANCHORS on silicon, so its
## executor-validation gate can pass (needs a board; anchors are then re-used
## by every later plain `prove-all` on the same OUT tree)
prove-all-anchor:
	python3 prove_all.py --all --allow-hardware --out $(OUT) --jobs $(JOBS)

## re-prove every op from scratch (ignore cached verdicts)
prove-all-force:
	python3 prove_all.py --all --force --out $(OUT) --jobs $(JOBS)

## guard routing + census math (fast unit + live 3-op end-to-end)
prove-all-selftest:
	python3 selftest_prove_all.py

## guard the 32-chip shard arithmetic + combiner fail-closed paths (no hardware)
shard-selftest:
	python3 selftest_galaxy_shard.py

## print the data-driven routing manifest census (no proving)
prove-all-manifest:
	@python3 -c "import prove_all as p; b=p.load_board(); m=p.load_manifest(b); \
from collections import Counter; c=Counter(r['engine'] for r in m.values()); \
print('op set == board:', set(m)==set(b), '(%d ops)'%len(m)); print('engines:', dict(c))"
