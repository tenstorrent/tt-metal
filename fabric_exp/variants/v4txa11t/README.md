# v4txa11t: bubble-rule injection threshold sweep (study, 2026-11-03)

v4txa11 plus one switch. An injection channel on a deadlock-avoiding VC sends only with **T** free far slots. That is
`SenderChannel<CH>::remote_free_slots_to_send` in `router_sender.hpp`. The channels are the worker and, on a torus, the
turns.
- T comes from `TT_FABRIC_EXP_DRISC_DELAY` (named arg `EXP_DRISC_DELAY`), as v4rxg3it passed its arms; there is no host
  change. Values 2..62 (widened from 2..6, then 2..16, on 2026-11-04 for the slot-depth jobs), and below the far
  receiver's slot count on every injection channel (static_assert in `send`). Unset or 0 means 2, the rule as shipped.
- The in-ring channel's 1 does not change.

Unchanged on purpose:
- `BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS` stays 2. So do the mux-extension kernels' own copies
  (`fabric_router_mux_extension.cpp`, `fabric_router_udm_mux_extension.cpp`) and the host's
  `bubble_flow_control_protected_receiver_min_slots` (`fabric_builder_config.hpp`). No row in this sweep uses mux mode.
- The far receivers hold T: 32 slots on the 1D ring and 16 on the torus with default slots; larger receivers come
  from TT_FABRIC_EXP_SLOTS_1D.

With the switch off the router images are byte-identical to v4txa11's. This README is not a source file:
offline_build.resolve and swap.py skip it.
