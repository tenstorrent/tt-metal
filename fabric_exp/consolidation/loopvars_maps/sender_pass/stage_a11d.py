#!/usr/bin/env python3
"""stage_a11d.py: v4txa11d = v4txa11 + v4bdd's study delay hooks (V4_BOUNDS.md; user go 2026-11-03), at the same
places, for the one-side delay A/Bs on the final router (bnd_delay_runs.sh BND_VARIANT=v4txa11d). Not for adoption.
  TT_FABRIC_EXP_TX_DELAY     a spin after each send (sender::send, after the packets-received update)
  TT_FABRIC_EXP_RX_DELAY     a spin after each packet handled (receiver::handle, after retire)
  TT_FABRIC_EXP_DRISC_DELAY  low 16 bits: a spin at the end of each sender pass; high 16 bits: each receiver pass
                             (sender::pass / receiver::pass, which a one-ERISC router runs as its two halves)
Every hook is `if constexpr (N > 0)`: at 0 each image must be byte-identical to v4txa11's (bnd_loadcmp.py, full
coverage at -O3; the *_delay / *_passdelay cases build with their delays and differ by design)."""
import os
import shutil
import sys

V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4txa11", f"{V}/v4txa11d"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
R = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router"


def edit(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, (path, s.count(old), old[:90])
    open(path, "w").write(s.replace(old, new))


# the constants, where v4bdd put them (after the host signal args)
edit(f"{dst}/fabric_erisc_router_ct_args.hpp",
     'constexpr uint32_t edm_channels_mask = wait_for_host_signal ? NAMED_CT_ARG("EDM_CHANNELS_MASK") : 0;\n',
     'constexpr uint32_t edm_channels_mask = wait_for_host_signal ? NAMED_CT_ARG("EDM_CHANNELS_MASK") : 0;\n'
     '\n'
     '// Study instruments (variants/v4txa11d = v4txa11 + v4bdd\'s hooks, V4_BOUNDS.md): fixed per-packet spins on the send\n'
     '// path (TT_FABRIC_EXP_TX_DELAY) and the receive path (TT_FABRIC_EXP_RX_DELAY), for measuring which side a row is\n'
     '// sensitive to; 0 compiles them out.\n'
     'constexpr uint32_t fabric_exp_tx_delay = NAMED_CT_ARG("EXP_TX_DELAY");\n'
     'constexpr uint32_t fabric_exp_rx_delay = NAMED_CT_ARG("EXP_RX_DELAY");\n'
     '// Per-pass spins (V4_PROFILE.md §6): TT_FABRIC_EXP_DRISC_DELAY, reused because v4 has no DRISC offload: its low 16\n'
     '// bits after each sender pass, its high 16 bits after each receiver pass; 0 compiles them out.\n'
     'constexpr uint32_t fabric_exp_sender_pass_delay = NAMED_CT_ARG("EXP_DRISC_DELAY") & 0xFFFFu;\n'
     'constexpr uint32_t fabric_exp_receiver_pass_delay = NAMED_CT_ARG("EXP_DRISC_DELAY") >> 16;\n')

# fabric_experiment_spin, declared before the router headers that name it
edit(f"{dst}/fabric_erisc_router_v4.cpp",
     '#include "tt_metal/fabric/hw/inc/edm_fabric/router/router_init.hpp"\n',
     '#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_phase_profiler.hpp"  // fabric_experiment_spin (study delays)\n'
     '#include "tt_metal/fabric/hw/inc/edm_fabric/router/router_init.hpp"\n')

# TX: after each send, after the packets-received update (v4bdd: after sent++, before telemetry)
edit(f"{R}/router_sender.hpp",
     '    remote_update_ptr_val<RemoteReceiverChannel<VC>::packets_received_stream, sender_txq_id>(1);\n'
     '    sent++;\n',
     '    remote_update_ptr_val<RemoteReceiverChannel<VC>::packets_received_stream, sender_txq_id>(1);\n'
     '    sent++;\n'
     '    // Study instrument (V4_BOUNDS.md): a fixed spin per packet sent (TT_FABRIC_EXP_TX_DELAY), so a row\'s sensitivity\n'
     '    // to the sender side\'s cost can be measured; compiled out at 0.\n'
     '    if constexpr (fabric_exp_tx_delay > 0) {\n'
     '        fabric_experiment_spin<fabric_exp_tx_delay>();\n'
     '    }\n')

# sender pass: at its end
edit(f"{R}/router_sender.hpp",
     '    visit_vc<2>(loop_copy, states, holding, channels_in_flight, send_yielded);\n}\n',
     '    visit_vc<2>(loop_copy, states, holding, channels_in_flight, send_yielded);\n'
     '    // Study instrument (V4_PROFILE.md §6): a fixed spin per sender pass; compiled out at 0.\n'
     '    if constexpr (fabric_exp_sender_pass_delay > 0) {\n'
     '        fabric_experiment_spin<fabric_exp_sender_pass_delay>();\n'
     '    }\n'
     '}\n')

# RX: after each packet handled, after retire
edit(f"{R}/router_receiver.hpp",
     '    retire<VC>(progress, sources, source_channel(packet.src_ch_id));\n    return true;\n',
     '    retire<VC>(progress, sources, source_channel(packet.src_ch_id));\n'
     '    // Study instrument (V4_BOUNDS.md): a fixed spin per packet received (TT_FABRIC_EXP_RX_DELAY); compiled out at 0.\n'
     '    if constexpr (fabric_exp_rx_delay > 0) {\n'
     '        fabric_experiment_spin<fabric_exp_rx_delay>();\n'
     '    }\n'
     '    return true;\n')

# receiver pass: at its end, after the credit frame
edit(f"{R}/router_receiver.hpp",
     '    if (credit_frame_due) {\n'
     '        credit_frame_due = !send_credit_frame(credits, txq_status);\n'
     '    }\n'
     '}\n',
     '    if (credit_frame_due) {\n'
     '        credit_frame_due = !send_credit_frame(credits, txq_status);\n'
     '    }\n'
     '    // Study instrument (V4_PROFILE.md §6): a fixed spin per receiver pass; compiled out at 0.\n'
     '    if constexpr (fabric_exp_receiver_pass_delay > 0) {\n'
     '        fabric_experiment_spin<fabric_exp_receiver_pass_delay>();\n'
     '    }\n'
     '}\n')
print(f"wrote {dst}")
