#!/bin/bash
# Blocks until both build and venv are complete.
# If either background job failed, exits cleanly so Claude
# can rebuild on its own and tests will work.
while [ ! -f "/localdev/vignjatijevic/2026_04_14/2132_vignjatijevic_test-softcap-strict-metrics/clones/softcap_sfpu_gen_1_run1/tt-metal/.build_complete" ]; do
    if [ -f "/localdev/vignjatijevic/2026_04_14/2132_vignjatijevic_test-softcap-strict-metrics/clones/softcap_sfpu_gen_1_run1/tt-metal/.infra_failed" ]; then
        echo "NOTE: Background build failed. Proceeding (agent may rebuild)." >&2
        break
    fi
    sleep 5
done
while [ ! -f "/localdev/vignjatijevic/2026_04_14/2132_vignjatijevic_test-softcap-strict-metrics/clones/softcap_sfpu_gen_1_run1/tt-metal/.venv_complete" ]; do
    if [ -f "/localdev/vignjatijevic/2026_04_14/2132_vignjatijevic_test-softcap-strict-metrics/clones/softcap_sfpu_gen_1_run1/tt-metal/.infra_failed" ]; then
        echo "NOTE: Background venv failed. Proceeding (agent may recreate)." >&2
        break
    fi
    sleep 5
done
