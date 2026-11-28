# DRAM counter check

`dram_validate.py` runs an example with a known DRAM traffic with `TT_METAL_PROFILE_DRAM_COUNTERS` off and on, and checks
that the option changes nothing but the logs and that the measured bytes match. Run it on an idle card nothing else has
open; `--device` picks the card and the example sees only that one.
