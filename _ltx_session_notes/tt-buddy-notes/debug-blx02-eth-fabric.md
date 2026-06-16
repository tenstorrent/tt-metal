# debug-blx02-eth-fabric

## Debug — 4x8 mesh-open fails: two inter-chip ETH links physically down (Dev24, Dev25)
**2026-06-15 20:48 UTC / 13:48 PT** · `tt-metal@6182a3b34c (pinned 6182a3b deploy /home/sulphur)`

**Mechanism:** tt-triage (check_eth_status, --dev=all, push-through exec; device idle, no live inspector)
**Verdict:** Hardware fabric fault — two Ethernet links won't train up. A galaxy soft/IPMI device reset does NOT recover them; needs a full power-cycle / physical attention. Not software, not Sulphur.

**Signals:**
- Device 24: eth [4-1 (e0,5)]: rx link is not up: Down — retrain count 3
- Device 25: eth [15-1 (e0,3)]: rx link is not up: Down — retrain count 3
- Surfaces at mesh open as: Device N timed out waiting for active ethernet core 27-25 (llrt.cpp:566). Seen on Device 0 and Device 24.

**Context:** LTX/Sulphur bh_4x8sp1tp0_linear jobs fail at setup in 26-57s before any weight load. Two MCP device resets cleared it for exactly one run earlier, then it re-wedged — consistent with links that re-train-fail rather than a soft state. Heavy concurrent agent churn (audio-hang debugging, many killed jobs) likely aggravated it.

**Next step:** escalate to blx02 owner / cold power-cycle affected trays (Dev24, Dev25). A soft reset is insufficient. Once links are up, the populated cache makes sulphur a fast cache-hit and the per-stage load logs render.

**Artifacts:**
- $HOME/.tt-buddy/triage/20260615T204734Z-eth2/triage_summary.txt
- $HOME/.tt-buddy/triage/20260615T204734Z-eth2/triage_output.txt
