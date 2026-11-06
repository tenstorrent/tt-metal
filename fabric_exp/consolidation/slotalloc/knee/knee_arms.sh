# Slot-need sweep arms (SLOT_ALLOCATOR.md section 12), one per line: SET ARM KIND SUITE ROW [ENV]
#   ENV (optional): extra environment for the arm, NAME=VALUE pairs joined by ";" (e.g. the completion-group switch)
#   KIND 2d | 1d | vc1 (P150 2x2 split, two ranks) | val (data-checked, default slots) for P150;
#        g2d | gvc1 (Galaxy 2x2 split) | gvc1x44 (Galaxy 4x4 split) | gx44san (its data-checked sanity) for Galaxy
#   SUITE knee/yamls/<SUITE>.yaml; ROW the explicit row (W,F,R or W,F0,R0,F1,R1; "-" = default slots)
# Held channels: off the rows' path at the minimum 2, otherwise above their need (section 12's table).
knee_arms() {
    local w r f
    # A: P150 4352 B, 2D. A1 worker (F 2 off path, R 32), A2 receiver (W 24, F 8), A3 forwarding (W 20, R 16).
    for w in 4 8 12 16 20 24 32; do echo "A w$w 2d p2w_4352 $w,2,32"; done
    for r in 4 8 16 32; do echo "A r$r 2d p2w_4352 24,8,$r"; done
    for f in 2 4 6 8 10 12 15; do echo "A f$f 2d p2f_4352 20,$f,16"; done
    echo "A f12w29 2d p2f_4352 29,12,16"
    # B: P150 4352 B, 1D. Worker (F 2, R 32), receiver (W 24, F 8), forwarding (W 24, R 32).
    for w in 4 8 12 16 20 24 32; do echo "B w$w 1d p1w_4352 $w,2,32"; done
    for r in 4 8 16 32; do echo "B r$r 1d p1w_4352 24,8,$r"; done
    for f in 2 4 8 12 16 24; do echo "B f$f 1d p1f_4352 24,$f,32"; done
    # C: P150 4352 B, VC1 2x2 split, VC0 held 14,7,16: forwarding at receiver 8, receiver at forwarding 6.
    for f in 2 4 6 7; do echo "C v${f}r8 vc1 pvc1_4352 14,7,16,$f,8"; done
    for r in 2 4; do echo "C v6r$r vc1 pvc1_4352 14,7,16,6,$r"; done
    # C2 (user, 2026-11-04): the VC1 knee past C's range. At 3072 B slots (N 115) VC0 keeps its worker (20) and receiver
    # (16) knees and forwarding 11 (below its >12 need, the most the budget leaves): VC0 20,11,16 (69), VC1 room 46.
    # VC1 forwarding 4-12 at receiver 8; VC1 receiver 4, 8, 16 at forwarding 10; a VC0-check arm (forwarding 14) at 8 / 8.
    for f in 4 6 8 10 12; do echo "C2 v${f}r8 vc1 pvc1_3072 20,11,16,$f,8"; done
    for r in 4 16; do echo "C2 v10r$r vc1 pvc1_3072 20,11,16,10,$r"; done
    echo "C2 v8r8chk vc1 pvc1_3072 20,14,16,8,8"
    # D: P150 3072 B, 2D: worker and receiver, three points each (held as A).
    for w in 8 16 24; do echo "D w$w 2d p2w_3072 $w,2,32"; done
    for r in 8 16 32; do echo "D r$r 2d p2w_3072 24,8,$r"; done
    # D2 (proposed 2026-11-04, not run): settle whether the 2D worker knee moves with the slot size: the same depths
    # between 16 and 24 at 3072 B and 4352 B slots, one host (held as A1: forwarding 2, receiver 32).
    for w in 16 18 20 22 24; do echo "D2 w${w}_3k 2d p2w_3072 $w,2,32"; echo "D2 w${w}_4k 2d p2w_4352 $w,2,32"; done
    # E: P150 8704 B (N 41): data-checked 1-hop writes first, then 1D worker / receiver / forwarding, 2D worker /
    # receiver.
    for s in 4352 8704 15232; do echo "E val1d$s val pval1d_$s -"; echo "E val2d$s val pval2d_$s -"; done
    for w in 4 8 12 16 20; do echo "E w${w}_1d 1d p1w_8704 $w,2,16"; done
    for r in 4 8 16; do echo "E r${r}_1d 1d p1w_8704 16,2,$r"; done
    for f in 2 4 6 9; do echo "E f${f}_1d 1d p1f_8704 16,$f,16"; done
    for w in 4 8 12 16 19; do echo "E w${w}_2d 2d p2w_8704 $w,2,16"; done
    for r in 4 8 16; do echo "E r${r}_2d 2d p2w_8704 16,2,$r"; done
    # F: P150 15232 B (N 23), the squeeze: worker-, forwarding-, receiver-favouring rows and main's tier pick.
    echo "F sqw_2d 2d p2sq_15232 13,2,4"; echo "F sqf_2d 2d p2sq_15232 4,5,4"
    echo "F sqr_2d 2d p2sq_15232 9,2,8"; echo "F sqt_2d 2d p2sq_15232 7,4,4"
    echo "F sqw_1d 1d p1sq_15232 13,2,8"; echo "F sqf_1d 1d p1sq_15232 4,11,8"
    echo "F sqr_1d 1d p1sq_15232 4,3,16"; echo "F sqt_1d 1d p1sq_15232 8,7,8"
    # G3 (Galaxy C2, user 2026-11-04): VC1 knee at 3072 B slots (N 115), VC0 held 20,11,16 as the P150 C2 (worker and
    # receiver at their knees, forwarding 11 below its >12 need); both splits; the 4x4 split's data-checked sanity first.
    echo "G3 x44san3 gx44san gvc1x44san_3072 20,11,16,10,8"
    for p in "" x44; do
        su=gvc1_3072; k=gvc1; [ -n "$p" ] && { su=gvc1x44_3072; k=gvc1x44; }
        for f in 4 6 8 10 12; do echo "G3 ${p}v${f}r8 $k $su 20,11,16,$f,8"; done
        for r in 4 16; do echo "G3 ${p}v10r$r $k $su 20,11,16,10,$r"; done
        echo "G3 ${p}v8r8chk $k $su 20,14,16,8,8"
    done
    # Q: the composition test (user, 2026-11-04; section 12.7), P150 2D, 4352 B slots. The requested "full" 20,16,16
    # needs 84 of 81 slots; forwarding 15 is the most that fits with worker 20 and receiver 16, so every arm whose
    # forwarding is "full" uses 15 (off the 1-hop rows' path; on the 3-hop / turn rows it is below forwarding's own need).
    echo "Q valq val pval2d_4352 -"
    echo "Q full 2d qall_4352 20,15,16"
    echo "Q w10 2d q1h_4352 10,15,16"
    echo "Q r8 2d q1h_4352 20,15,8"
    echo "Q w10r8 2d q1h_4352 10,15,8"
    echo "Q w14r4 2d q1h_4352 14,15,4"
    echo "Q f8 2d q3h_4352 20,8,16"
    echo "Q f8r8 2d q3h_4352 20,8,8"
    echo "Q w10f8r8 2d qall_4352 10,8,8"
    # CG: the completion-group A/B (user, 2026-11-04; section 12.8). Receiver groups: 8-15 slots -> 2, >= 16 -> 4.
    echo "CG valcg val pval2d_4352 -"
    for r in 8 12 16 20 24; do echo "CG w10r$r 2d q1h_4352 10,15,$r"; done
    for r in 12 16; do echo "CG w20r$r 2d q1h_4352 20,15,$r"; done
    for r in 12 16; do echo "CG f8r$r 2d q3h_4352 20,8,$r"; done
    # XO: the group-size crossover (user, 2026-11-04; section 12.9). Worker 8-24 at receiver 12 (G2) and 16 (G4),
    # forwarding 15 (24 with receiver 16 needs 85 of 81: forwarding 13 there, off the 1-hop path); forwarding 6-15 at
    # receiver 12 and 16, worker 20.
    echo "XO valxo val pval2d_4352 -"
    for w in 8 12 16 20 24; do
        echo "XO w${w}r12 2d x1h_4352 $w,15,12"
        f=15; [ $((w + 45 + 16)) -gt 81 ] && f=$(((81 - w - 16) / 3))
        echo "XO w${w}r16 2d x1h_4352 $w,$f,16"
    done
    for f in 6 8 10 12 15; do echo "XO f${f}r12 2d q3h_4352 20,$f,12"; echo "XO f${f}r16 2d q3h_4352 20,$f,16"; done
    # M24 / M42 / MG: completion group size against k far sender channels into one receiver (user, 2026-11-04;
    # section 12.10). v4txa11id (KNEE_VARIANT), 3072 B slots (the merge yamls), TT_FABRIC_EXP_COMPLETION_GROUP = 1 / 2 / 4
    # at senders squeezed (worker / forwarding 8 / 8, 12 / 12) and not (20 / 16), receiver 16 and 32. The data-checked
    # merge rows first, at the default slots. M42 runs the P150 as a 4x2 mesh (study descriptor).
    local mgd42="TT_MESH_GRAPH_DESC_PATH=/home/aagarwal/tt-metal/fabric_exp/consolidation/slotalloc/knee/merge/p150_x8_4x2_mesh_mgd.textproto"
    for set in M24 M42 MG; do
        local shape kind ex0=""
        case $set in M24) shape=p2x4; kind=2d ;; M42) shape=p4x2; kind=2d; ex0=$mgd42 ;; MG) shape=g8x4; kind=g2d ;; esac
        echo "$set val${shape} $([ $set = MG ] && echo gmval || echo val) merge_${shape}_val - ${ex0:--}"
        for wf in 8,8 12,12 20,16; do
            for r in 16 32; do
                for g in 1 2 4; do
                    local ex="TT_FABRIC_EXP_COMPLETION_GROUP=$g${ex0:+;$ex0}"
                    echo "$set w${wf%,*}f${wf#*,}r${r}g$g $kind merge_${shape}_bench ${wf%,*},${wf#*,},$r $ex"
                done
            done
        done
    done
    # L1: the 1D large-slot check (user, 2026-11-04; section 12.11): P150 1D at 8704 B slots, the default 1D row
    # 13,12,16, TT_FABRIC_EXP_COMPLETION_GROUP = 1 / 2 / 4 (v4txa11id); a lone worker, a lone pass-through and both
    # merging into one receiver. The data-checked rows first, at the default slots.
    echo "L1 valp1d8k val merge_p1d_8704_val - -"
    for g in 1 2 4; do echo "L1 w13f12r16g$g 1d merge_p1d_8704_bench 13,12,16 TT_FABRIC_EXP_COMPLETION_GROUP=$g"; done
    # Q2: the singles the F/R and three-way pairs need on the 3-hop / turn rows (receiver 8 alone, worker 10 alone).
    echo "Q2 r8q3 2d q3h_4352 20,15,8"
    echo "Q2 w10q3 2d q3h_4352 10,15,16"
    # G1: Galaxy 4352 B, 2D, as A.
    for w in 4 8 12 16 20 24 32; do echo "G1 w$w g2d g2w_4352 $w,2,32"; done
    for r in 4 8 16 32; do echo "G1 r$r g2d g2w_4352 24,8,$r"; done
    for f in 2 4 6 8 10 12 15; do echo "G1 f$f g2d g2f_4352 20,$f,16"; done
    echo "G1 f12w29 g2d g2f_4352 29,12,16"
    # G2: Galaxy 4352 B, VC1: the 4x4 split's data-checked sanity first (default slots), then the 2x2 and 4x4 splits
    # as C; on the 4x4 split a VC0-check arm (more VC0 at forwarding 6 / receiver 8).
    echo "G2 x44san gx44san gvc1x44san_4352 -"
    for f in 2 4 6 7; do echo "G2 v${f}r8 gvc1 gvc1_4352 14,7,16,$f,8"; done
    for r in 2 4; do echo "G2 v6r$r gvc1 gvc1_4352 14,7,16,6,$r"; done
    for f in 2 4 6 7; do echo "G2 x44v${f}r8 gvc1x44 gvc1x44_4352 14,7,16,$f,8"; done
    for r in 2 4; do echo "G2 x44v6r$r gvc1x44 gvc1x44_4352 14,7,16,6,$r"; done
    echo "G2 x44chk gvc1x44 gvc1x44_4352 18,7,16,6,8"
}
