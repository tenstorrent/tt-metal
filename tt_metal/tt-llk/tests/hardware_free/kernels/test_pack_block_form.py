# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Python mirror of the form llk_pack_block takes per cell (tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h), checked against the header.

A block takes the per-tile pack (main's code, T), the plain or closed block that programs the output (B) or the block with the W offset (W). The form tables hold one row per output
class (tiles of 16x16, 16x32, 8x32, 4x32 and 1x32, Bfp8_b and Bfp8, Bfp4_b and Bfp2_b, Bfp4 and Bfp2, 4 byte datums, the other plain formats) and one column per DEST width and sync, from the tt-llk crossover sweeps.
The first block of an output takes the init or the reconfigure table, the blocks that continue it the next table. A kernel with a 32 bit CB next to a 32 bit DEST takes the tables of all sweeps, the others the _DC tables of the datacopy
family. Behind matmul math the cells the matmul sweeps read slower are forbidden and, with the K given, the cells where the W offset reads faster take it (masks below).
"""

import pathlib
import re

API = pathlib.Path(__file__).resolve().parents[4] / "hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h"
ALT = pathlib.Path(__file__).resolve().parents[2] / "sources/pack_block_alt_test.cpp"
CLASSES = ("s16x16", "s16x32", "s8x32", "s4x32", "s1x32", "bfp8", "bfp4", "bfpa", "wide", "narrow", "f16b", "f32")

INIT_FORM = {
    "s16x16": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s16x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s8x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s4x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s1x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "bfp8": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"],
    "bfp4": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"],
    "bfpa": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TTBTBTTTTTTTTTTT"],
    "wide": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "narrow": ["TBBBBBBBBBBBBBBB", "TBBTTTTTTTTTTTTT", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "f16b": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "f32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
}
RECONF_FORM = {
    "s16x16": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s16x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s8x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s4x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s1x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "bfp8": ["TBBBBBBBBBBBBBBB", "TBTBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"],
    "bfp4": ["TBBBBBBBBBBBBBBB", "TBBTTTTTTTTTTTTT", "TBBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"],
    "bfpa": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "wide": ["BBBBBBBBBBBBBBBB", "BBTTTTTTTTTTTTTT", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"],
    "narrow": ["BBBBBBBBBBBBBBBB", "BBTTTTTTTTTTTTTT", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"],
    "f16b": ["BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"],
    "f32": ["BBBBBBBBBBBBBBBB", "BBTTTTTTTTTTTTTT", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"],
}
NEXT_FORM = {
    "s16x16": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBBWBBBBBBBB", "TWWBBBBBBBBBBBBB"],
    "s16x32": ["TWWWBWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBBBBBBWWWWW", "TWWBBBBBBBBBBBBB"],
    "s8x32": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBBBBBBBBBB"],
    "s4x32": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBBBBBBBBBB"],
    "s1x32": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBWWWWWWWWW"],
    "bfp8": ["TWBBBBWWWWWWWWWW", "TBWTTTTTTTTTTTTT", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"],
    "bfp4": ["TWBBWWWWWWWWWWWW", "TBBBBBBBBBBBBBBB", "TWWWWWWWWWWWWWWW", "TBWWWWWWWWWWWWWW"],
    "bfpa": ["TWBBWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"],
    "wide": ["BWWWBBBBBBBBBBBB", "TBWBBBBBBBBBBBBB", "BWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"],
    "narrow": ["BBBWWWWWWWWWWWWW", "TBWWWWWWWWWWWWWW", "BBBWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"],
    "f16b": ["WWWWWWWWWWWWWWWW", "TBBBBBBBBBBBBBBB", "WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"],
    "f32": ["WWWWBBBBBBBBBBBB", "TBWBBBBBBBBBBBBB", "WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"],
}
INIT_FORM_DC = {
    "s16x16": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s16x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s8x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s4x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s1x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "bfp8": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"],
    "bfp4": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"],
    "bfpa": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TTBTBTTTTTTTTTTT"],
    "wide": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "narrow": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "f16b": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "f32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
}
RECONF_FORM_DC = {
    "s16x16": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s16x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s8x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s4x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s1x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "bfp8": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "bfp4": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "bfpa": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "wide": ["BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"],
    "narrow": ["BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"],
    "f16b": ["BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"],
    "f32": ["BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"],
}
NEXT_FORM_DC = {
    "s16x16": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBBWBBBBBBBB", "TWWBBBBBBBBBBBBB"],
    "s16x32": ["TWWWBWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBBBBBBWWWWW", "TWWBBBBBBBBBBBBB"],
    "s8x32": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBBBBBBBBBB"],
    "s4x32": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBBBBBBBBBB"],
    "s1x32": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBWWWWWWWWW"],
    "bfp8": ["TWBBBBWWWWWWWWWW", "TWBBBBBBBBBBBBBB", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"],
    "bfp4": ["TWBBWWWWWWWWWWWW", "TWBBBBBBBBBBBBBB", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"],
    "bfpa": ["TWBBWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"],
    "wide": ["BWWWBBBBBBBBBBBB", "TWWWWWWWWWWWWWWW", "BWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"],
    "narrow": ["BBBWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "BBBWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"],
    "f16b": ["WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"],
    "f32": ["WWWWBBBBBBBBBBBB", "WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"],
}
# Matmul masks: bit column * 16 + n - 1 at n tiles. NO_BLOCK and NO_W [family][group][state][class] (state init, reconf, next), TAKE_* [family][group - 1][class]: family 0 datacopy, 1 32 bit input; groups LoFi K not given, LoFi K 1, LoFi K above 1, HiFi2 to HiFi4 (TAKE_W without the first group).
MM_NO_BLOCK = [[[[9570295237181474, 562958544011482, 562958543486978, 562958543487010, 562958543486978, 562958544011434, 562958544011306, 6192458077962474, 4294967425, 281479271743649, 281479271743649, 4294967425], [9570295237050400, 562958543880408, 562958543355904, 562958543355936, 562958543355904, 562958544011450, 562958544011434, 562958543487210, 201, 281474976776353, 281474976776353, 201], [9007336693694496, 524504, 0, 32, 0, 562958544011434, 562958544011306, 562958543487210, 4294967425, 281479272792225, 281479271743649, 4294967425]], [[0, 0, 0, 0, 0, 0, 0, 8589934592, 0, 0, 0, 0], [0, 0, 0, 0, 0, 562958543486978, 562958543486978, 562958543486978, 0, 0, 0, 0], [0, 0, 0, 0, 0, 131074, 131072, 2, 1, 65536, 65537, 1]], [[9007336693694496, 524504, 0, 32, 0, 524456, 524328, 8590196968, 128, 281474976710816, 4294967456, 128], [9007336693694496, 524504, 0, 32, 0, 8590590138, 562949954076842, 8590065898, 200, 160, 160, 200], [9007336693694496, 524504, 0, 32, 0, 524456, 655402, 234, 128, 1048736, 160, 128]], [[562958544273662, 562958544011434, 562958544011498, 562958544273614, 562958544273646, 8444309431779582, 3940675444146278, 6192475258355814, 12884902117, 844429225164929, 844429225164929, 12884902117], [562958544273662, 562958544011498, 562958544011498, 562958544273614, 562958544273646, 1688858450854122, 562958543487206, 562958543487078, 125, 281474976776433, 281474976776433, 125], [917758, 917678, 655594, 917726, 917758, 562958543487090, 562958544273454, 562958544273454, 4294967405, 281479271874675, 281479271874675, 4294967405]], [[562958543486978, 562958543486978, 562958543486978, 562958543486978, 562958543486978, 562958543486978, 562958543486978, 6192458077962242, 4294967425, 281479271743649, 281479271743489, 4294967297], [562958543355904, 562958543355904, 562958543355904, 562958543355904, 562958543355904, 562958543486978, 562958543486978, 562958543486978, 129, 281474976776353, 281474976776193, 1], [0, 0, 0, 0, 0, 562958543486978, 562958543486978, 562958543486978, 4294967425, 281479271743649, 281479271743489, 4294967297]], [[562958543486978, 562958543486978, 562958543487170, 562958543486978, 562958543486978, 8444309431779582, 3940675444146246, 6192475258355782, 12884902117, 844429225164929, 844429225164929, 12884902053], [562958543355904, 562958543356096, 562958543355904, 562958543356032, 562958543356032, 1688858450854026, 562958543487206, 562958543487078, 125, 281474976776433, 281474976776433, 125], [0, 262148, 0, 16, 16, 562958543487090, 562958544273454, 562958544273454, 4294967405, 281479271874675, 281479271874675, 4294967341]]], [[[9570286647771168, 917720, 562949953552384, 562949953814560, 562949954338816, 8444249302237352, 3940649674604584, 6192449488552168, 281474977300608, 281474977693856, 281474977693856, 281474977300608], [9007336693825568, 786648, 0, 562949953683488, 562949954207744, 3940649674866856, 3940649674866728, 562949954076904, 589952, 281474977038496, 281474977038496, 589952], [9007336693694496, 786648, 0, 262176, 786432, 562949954338984, 562949954076712, 562949954339048, 281474977562752, 281474977300640, 281474976776352, 281474977562752]], [[0, 0, 0, 0, 0, 524288, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 562949954076672, 562949953552384, 562949953552384, 0, 0, 0, 0], [0, 0, 0, 0, 0, 655360, 131072, 0, 65536, 0, 65536, 65536]], [[9007336693694496, 786648, 0, 262176, 786432, 917672, 524328, 262376, 524416, 160, 160, 524416], [9007336693694496, 786648, 0, 262176, 786432, 917672, 562949954207784, 131304, 524416, 160, 160, 524416], [9007336693694496, 786648, 0, 262176, 786432, 917672, 524328, 524520, 524416, 160, 160, 524416]], [[562949954339070, 917674, 562949954339050, 562949954339022, 562949954339054, 8444249302237282, 3940649674866722, 6192449488551970, 1970324837433408, 844424931115008, 844424931115008, 1970324837433408], [917758, 917674, 917738, 562949954339022, 562949954339054, 3940649674080354, 3940649674342434, 562949954338850, 852032, 281474977038336, 281474977038336, 852032], [917758, 917674, 917738, 917710, 917742, 562949953814626, 562949954338850, 562949954338850, 281474977562688, 281474977693696, 281474977169408, 281474977562688]], [[563087393030176, 131288, 562949953552384, 562949953552416, 562949953552384, 8444249301450920, 3940649674604584, 6192449488552168, 281474976776320, 281474977693856, 281474977693856, 281474976776320], [137439084576, 216, 0, 562949953421344, 562949953421312, 3940649674080424, 3940649674342440, 562949954076904, 589952, 281474977038496, 281474977038496, 589952], [137438953504, 216, 0, 32, 0, 562949953552552, 562949953552424, 562949954339048, 281474977562752, 281474977300640, 281474976776352, 281474977562752]], [[562949954076926, 131242, 562949953552618, 562949953552590, 562949953552622, 8444249302237282, 3940649674866722, 6192449488551970, 1970324837433408, 844424931115008, 844424931115008, 1970324837433408], [131326, 170, 234, 562949953421518, 562949953421550, 3940649674080354, 3940649674342434, 562949954338850, 852032, 281474977038336, 281474977038336, 852032], [254, 262314, 234, 206, 238, 562949953814626, 562949954338850, 562949954338850, 281474977562688, 281474977693696, 281474977169408, 281474977562688]]]]
MM_NO_W = [[[[3940709803622434, 562958543487122, 562958543486978, 562958543487074, 562958543487106, 562992903749818, 562992903749674, 6192458077962474, 4294967297, 281479271743649, 281479271743649, 4294967297], [3940709803491360, 562958543356048, 562958543355904, 562958543356000, 562958543356032, 562992903749818, 562992903749674, 562958543487210, 1, 281474976776353, 281474976776353, 1], [3940709804408878, 1125899907760286, 1689133329022990, 1688875630461030, 1688875630461062, 562992903749818, 562992903749738, 8590852330, 4294967391, 281479273709735, 281479272661167, 4294967391]], [[562958543355904, 0, 0, 0, 562949953421312, 0, 0, 0, 0, 0, 0, 0], [562958543355904, 0, 0, 0, 562949953421312, 0, 0, 0, 0, 0, 0, 0], [562958543355904, 0, 0, 0, 562949953421312, 131074, 131074, 131074, 1, 65536, 65537, 1]], [[3940709803491360, 144, 0, 96, 562949953421440, 42950197432, 562992903618600, 8589934824, 0, 160, 160, 0], [3940709803491360, 144, 0, 96, 562949953421440, 42950197432, 562992903618600, 8589934824, 0, 160, 160, 0], [3940709803491360, 144, 0, 96, 562949953421440, 42950197432, 562992903749738, 8590721258, 89, 281474978349216, 281479272267945, 89]], [[562958544273646, 562958544273582, 562958544011498, 562958544273614, 562958544273582, 8444309431779582, 3940675444146278, 6192475258355814, 12884902117, 844429225296003, 844429225296003, 12884902117], [562958544273646, 562958544273646, 562958544011498, 562958544273614, 562958544273582, 1688858450854122, 562958543487206, 562958543487078, 125, 281474976907507, 281474976907507, 125], [1688858451116270, 1125899907760302, 1689133329023214, 1688875630985438, 1688875630985406, 562958544273662, 562958544273454, 8590852142, 4294967423, 281479272661007, 281479272661007, 4294967423]], [[562958543486978, 562958543486978, 562958543486978, 562958543486978, 562958543486978, 562958543486978, 562958543486978, 6192458077962242, 4294967297, 281479271743649, 281479271743489, 4294967297], [562958543355904, 562958543355904, 562958543355904, 562958543355904, 562958543355904, 562958543486978, 562958543486978, 562958543486978, 1, 281474976776353, 281474976776193, 1], [1688858451116046, 1125899907760142, 1689133329022990, 1688875630460934, 1688875630460934, 562958543486978, 562958543486978, 8590327810, 4294967311, 281479272136871, 281479272136711, 4294967311]], [[562958543486978, 562958543486978, 562958543487170, 562958543486978, 562958543486978, 8444309431779582, 3940675444146246, 6192475258355782, 12884902117, 844429225296003, 844429225164929, 12884902053], [562958543355904, 562958543356096, 562958543355904, 562958543356032, 562958543356032, 1688858450854026, 562958543487206, 562958543487078, 125, 281474976907507, 281474976776433, 125], [1688858451116046, 1125899907760142, 1689133329023118, 1688875630460950, 1688875630460950, 562958544273662, 562958544273454, 8590852142, 4294967423, 281479272661007, 281479272661007, 4294967359]]], [[[3940709804146720, 917648, 562949953814528, 562949953552480, 562949954076800, 8444292251648184, 3940692624277544, 6192458078486760, 281474977431552, 281474977693856, 281474977693856, 281474977431552], [3940709803622432, 786576, 262144, 562949953421408, 562949953945728, 3940692624277688, 3940692624539688, 562958544011496, 720896, 281474977169568, 281474977169568, 720896], [3940709803884576, 1125899907760272, 1125899907760128, 1125899907235936, 1688849861181568, 562992903749816, 562992903749672, 8590852328, 281474977693696, 281474977693856, 281474977693856, 281474977693696]], [[562958543355904, 0, 0, 0, 562949953421312, 655360, 562949954076672, 0, 0, 0, 0, 0], [562958543355904, 0, 0, 0, 562949953421312, 655360, 562949954076672, 0, 0, 0, 0, 0], [562958543355904, 0, 0, 0, 562949953421312, 655360, 562949954076672, 131072, 65536, 0, 65536, 65536]], [[3940709803491360, 786576, 262144, 96, 562949953945728, 42950328504, 562992903749672, 8589934824, 655360, 131232, 131232, 655360], [3940709803491360, 786576, 262144, 96, 562949953945728, 42950328504, 562992903749672, 8589934824, 655360, 131232, 131232, 655360], [3940709803491360, 786576, 262144, 96, 562949953945728, 42950328504, 562992903749672, 8590590184, 655360, 131232, 131232, 655360]], [[562949954339054, 917678, 562949954339050, 562949954339022, 562949954338990, 8444249302237282, 3940649674866722, 6192449488551970, 1970324837433408, 844424931115010, 844424931115010, 1970324837433408], [562949954339054, 917678, 917738, 562949954339022, 562949954338990, 3940649674866786, 3940649674342434, 562949954338850, 852032, 281474977169410, 281474977169410, 852032], [1688849861181678, 1125899907760302, 1125899907760362, 1125899907760334, 1688849861181614, 562949954338914, 562949954338850, 917538, 281474977693760, 281474977693698, 281474977693698, 281474977693760]], [[563010083618848, 131216, 562949953552384, 562949953552480, 562949953552512, 8444292251123896, 3940692624277544, 6192458078486760, 281474977431552, 281474977693856, 281474977693856, 281474976776192], [60129673248, 144, 0, 562949953421408, 562949953421440, 3940692623753400, 3940692624015400, 562958544011496, 720896, 281474977169568, 281474977038496, 589824], [1125960036778016, 1125899907498128, 1125899907760128, 1125899907235936, 1125899907235968, 562992903225528, 562992903225384, 8590852328, 281474977693696, 281474977693856, 281474977562784, 281474977693696]], [[562949954076910, 131246, 562949953552618, 562949953552590, 562949953552558, 8444249302237282, 3940649674866722, 6192449488551970, 1970324837433408, 844424931115010, 844424931115010, 1970324837433408], [131310, 174, 234, 562949953421518, 562949953421486, 3940649674080354, 3940649674342434, 562949954338850, 852032, 281474977169410, 281474977038338, 852032], [1125899907236078, 1125899907760302, 1125899907760362, 1125899907236046, 1125899907236014, 562949954338914, 562949954338850, 917538, 281474977693760, 281474977693698, 281474977693698, 281474977693760]]]]
MM_TAKE_W = [[[70086685085401088, 69814555957526544, 70086410207494144, 70086410207494144, 34057613188530176, 786492, 786444, 12, 0, 0, 0, 240], [9236035112271872, 69814555957526528, 70086410207494144, 70086410207494144, 34057613188530176, 262148, 0, 4, 0, 0, 0, 160], [70086685085401088, 69814555957526528, 70086410207494144, 70086410207494144, 34057613188530176, 786460, 0, 0, 0, 0, 0, 176]], [[70086685085401088, 69814555957526544, 70086410207494144, 70086410207494144, 34057613188530176, 60, 562949953683468, 12, 0, 0, 917504, 655600], [63279230640717824, 69814555957526528, 70086410207494144, 70086410207494144, 34057613188530176, 131076, 393216, 4, 0, 0, 786432, 240], [70086685085401088, 69814555957526528, 70086410207494144, 70086410207494144, 34057613188530176, 28, 562949953421312, 0, 0, 0, 262144, 655536]]]
MM_TAKE_B = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 524288, 0, 0, 0, 0, 0, 0]]]
FORM_NAMES = {"T": "per_tile", "B": "block", "W": "w_offset"}
STATE_INDEX = {"init": 0, "reconf": 1, "next": 2}
BFP_8 = ("Bfp8_b", "Bfp8")
BFP_4 = ("Bfp4_b", "Bfp2_b")
BFP_A = ("Bfp4", "Bfp2")
WIDE = ("Float32", "Int32", "UInt32", "Tf32")


def output_class(dst_format, num_faces=4, face_r_dim=16):
    if num_faces != 4 or face_r_dim != 16:
        if face_r_dim == 16:
            return "s16x16" if num_faces == 1 else "s16x32"
        return "s8x32" if face_r_dim >= 8 else ("s4x32" if face_r_dim >= 4 else "s1x32")
    if dst_format in BFP_A:
        return "bfpa"
    if dst_format in BFP_8:
        return "bfp8"
    if dst_format in BFP_4:
        return "bfp4"
    if dst_format == "Float16_b":
        return "f16b"
    if dst_format == "Float32":
        return "f32"
    return "wide" if dst_format in WIDE else "narrow"


def pack_block_form(ntiles, dst_format, state, fp32_dest=False, full_sync=False, hifi=False, num_faces=4, face_r_dim=16, matmul_kt=0, src32=False, after_matmul=True, u2d=True):
    """Form of one llk_pack_block call; state is init (first block after a pack init or another output), reconf (first block after a packer reconfigure) or next (the block continues the output).
    src32: a Float32, Int32 or UInt32 CB sits in the kernel's CB list (the output's own CB counts, for its unpack source format is its format).
    u2d: a CB of the kernel unpacks to DEST (LLK_UNPACK_TO_DEST_ANY); the form tables of the 32 bit input family need it, the masks do not.
    after_matmul: the math context byte of pack_ctx is matmul (a matmul init ran on the pack thread); without it and without the K the masks of the other contexts apply (groups 4, 5)."""
    if not fp32_dest and dst_format in ("Int8", "UInt8"):
        return FORM_NAMES["T"]
    family = 1 if (fp32_dest and (src32 or dst_format in ("Float32", "Int32", "UInt32"))) else 0
    table = {"init": (INIT_FORM_DC, INIT_FORM), "reconf": (RECONF_FORM_DC, RECONF_FORM), "next": (NEXT_FORM_DC, NEXT_FORM)}[state][family if u2d else 0]
    column = (2 if full_sync else 0) + (1 if fp32_dest else 0)
    cls = output_class(dst_format, num_faces, face_r_dim)
    row = CLASSES.index(cls)
    count = min(ntiles, 16) - 1
    bit = column * 16 + count
    form = table[cls][column][count]
    group = 3 if hifi else (0 if matmul_kt == 0 else (1 if matmul_kt == 1 else 2))
    mask_group = group if (matmul_kt != 0 or after_matmul) else (5 if hifi else 4)
    if form == "W" and (MM_NO_W[family][mask_group][STATE_INDEX[state]][row] >> bit) & 1:
        form = "B"
    if form == "B" and (MM_NO_BLOCK[family][mask_group][STATE_INDEX[state]][row] >> bit) & 1:
        form = "T"
    if matmul_kt != 0 and state == "next" and form != "W" and (MM_TAKE_W[family][group - 1][row] >> bit) & 1:
        form = "W"
    if matmul_kt != 0 and state == "next" and form == "T" and (MM_TAKE_B[family][group - 1][row] >> bit) & 1:
        form = "B"
    return FORM_NAMES[form]


def test_unpack_to_srca_takes_datacopy_tables():
    """A 32 bit CB without unpack to DEST (Float32 to SrcA) takes the datacopy rows: the Float32 input sweep to SrcA reads every block form faster than per tile."""
    cells = [(n, state, fmt) for n in range(1, 9) for state in ("init", "reconf", "next") for fmt in ("Float32", "Float16", "Bfp8_b")]
    assert any(pack_block_form(n, fmt, state, fp32_dest=True, src32=True, u2d=True) == "per_tile" and pack_block_form(n, fmt, state, fp32_dest=True, src32=True, u2d=False) != "per_tile" for n, state, fmt in cells)


def test_mirror_cells():
    assert pack_block_form(1, 'Float16_b', 'init') == 'per_tile'
    assert pack_block_form(2, 'Float16_b', 'init') == 'block'
    assert pack_block_form(2, 'Float16_b', 'next') == 'block'
    assert pack_block_form(4, 'Float16_b', 'next', fp32_dest=True) == 'block'
    assert pack_block_form(2, 'Bfp8_b', 'reconf') == 'per_tile'
    assert pack_block_form(3, 'Bfp8_b', 'init') == 'block'
    assert pack_block_form(8, 'Bfp8_b', 'next') == 'per_tile'
    assert pack_block_form(12, 'Bfp8_b', 'next', full_sync=True) == 'w_offset'
    assert pack_block_form(16, 'Bfp8_b', 'init', full_sync=True) == 'block'
    assert pack_block_form(12, 'Float32', 'next', full_sync=True) == 'w_offset'
    assert pack_block_form(16, 'Float16_b', 'reconf', full_sync=True) == 'block'
    assert pack_block_form(4, 'Int8', 'init') == 'per_tile'
    assert pack_block_form(4, 'Int8', 'init', fp32_dest=True) == 'block'
    assert pack_block_form(3, 'UInt8', 'next') == 'per_tile'
    assert pack_block_form(16, 'Fp8_e4m3', 'next', full_sync=True) == 'w_offset'
    assert pack_block_form(3, 'Tf32', 'next') == 'block'
    assert pack_block_form(4, 'Bfp8_b', 'next', matmul_kt=4) == 'per_tile'
    assert pack_block_form(4, 'Bfp8_b', 'next', matmul_kt=1) == 'w_offset'
    assert pack_block_form(4, 'Bfp8_b', 'next', hifi=True) == 'block'
    assert pack_block_form(4, 'Bfp8_b', 'next') == 'per_tile'
    assert pack_block_form(3, 'Float16_b', 'init', num_faces=2, face_r_dim=16) == 'block'
    assert pack_block_form(5, 'Float16_b', 'next', num_faces=2, face_r_dim=8) == 'w_offset'
    assert pack_block_form(3, 'Bfp8_b', 'next', fp32_dest=True) == 'block'
    assert pack_block_form(4, 'Bfp8_b', 'next', fp32_dest=True) == 'per_tile'
    assert pack_block_form(3, 'Bfp8_b', 'next', fp32_dest=True, src32=True) == 'w_offset'
    assert pack_block_form(4, 'Bfp8_b', 'next', fp32_dest=True, src32=True) == 'per_tile'
    assert pack_block_form(4, 'Bfp8_b', 'next', after_matmul=False) == 'block'
    assert pack_block_form(3, 'Float16_b', 'next', hifi=True, after_matmul=False) == 'block'
    assert pack_block_form(5, 'Float16_b', 'next', hifi=True, matmul_kt=0) == 'w_offset'
    assert pack_block_form(6, 'Bfp8_b', 'next', hifi=True, matmul_kt=0) == 'per_tile'
    assert pack_block_form(5, 'Float16_b', 'next', hifi=True, matmul_kt=1) == 'w_offset'
    assert pack_block_form(6, 'Bfp8_b', 'next', hifi=True, matmul_kt=1) == 'per_tile'
    assert pack_block_form(5, 'Float16_b', 'next', hifi=True, matmul_kt=4) == 'w_offset'
    assert pack_block_form(6, 'Bfp8_b', 'next', hifi=True, matmul_kt=4) == 'per_tile'


def _table(src, name):
    m = re.search(rf"constexpr char {name}\[12\]\[4\]\[17\] = \{{(.*?)\n\}};", src, re.S)
    assert m, name
    rows = re.findall(r"\{((?:\"[TBW]{16}\",? ?){4})\},\s*//\s*(.*)", m.group(1))
    return {CLASSES[i]: re.findall(r"\"([TBW]{16})\"", r) for i, (r, _) in enumerate(rows)}


def _masks(src, name):
    m = re.search(rf"constexpr std::uint64_t {name}((?:\[\d+\])+) = \{{(.*?)\n\}};", src, re.S)
    assert m, name
    dims = [int(d) for d in re.findall(r"\d+", m.group(1))]
    flat = [[int(x, 16) if x.startswith("0x") else int(x) for x in re.findall(r"0x[0-9A-F]{16}|\b0\b", g)] for g in re.findall(r"\{([^{}]*)\}", m.group(2))]
    for d in reversed(dims[1:-1]):
        flat = [flat[i:i + d] for i in range(0, len(flat), d)]
    return flat


def test_header_tables_match_mirror():
    src = API.read_text()
    assert _table(src, "llk_pack_init_form") == INIT_FORM
    assert _table(src, "llk_pack_reconf_form") == RECONF_FORM
    assert _table(src, "llk_pack_next_form") == NEXT_FORM
    assert _table(src, "llk_pack_init_form_dc") == INIT_FORM_DC
    assert _table(src, "llk_pack_reconf_form_dc") == RECONF_FORM_DC
    assert _table(src, "llk_pack_next_form_dc") == NEXT_FORM_DC
    assert _masks(src, "llk_pack_mm_no_block") == MM_NO_BLOCK
    assert _masks(src, "llk_pack_mm_no_w") == MM_NO_W
    assert _masks(src, "llk_pack_mm_take_w") == MM_TAKE_W
    assert _masks(src, "llk_pack_mm_take_b") == MM_TAKE_B


def test_header_logic_matches_mirror():
    src = " ".join(API.read_text().split())
    assert "const std::uint32_t column = (DST_SYNC_MODE == DstSync::SyncFull ? 2 : 0) + (is_fp32_dest_acc_en ? 1 : 0);" in src
    assert "const std::uint32_t count = (ntiles < 16 ? ntiles : 16) - 1;" in src
    assert "if (!is_fp32_dest_acc_en && llk_pack_dst_is_int8(output_id)) { return 'T'; }" in src
    assert "const std::uint32_t bit = column * 16 + count;" in src
    assert "constexpr std::uint32_t family = llk_pack_src32_family<is_fp32_dest_acc_en> ? 1 : 0;" in src
    assert "inline constexpr bool llk_pack_src32_family = is_fp32_dest_acc_en && (llk_pack_src32_mask(0) != 0 || llk_pack_src32_mask(32) != 0);" in src
    assert "constexpr std::uint32_t group = MATH_FIDELITY != MathFidelity::LoFi ? 3 : (matmul_kt == 0 ? 0 : (matmul_kt == 1 ? 1 : 2));" in src
    assert "constexpr std::uint32_t group_other = MATH_FIDELITY != MathFidelity::LoFi ? 5 : 4; const std::uint32_t mask_group = (matmul_kt != 0 || ctx != 0) ? group : group_other;" in src
    assert "const std::uint32_t state, const std::uint32_t ctx = llk_pack_context()) {" in src
    assert "PACK((llk_pack_matmul_context()));" in (API.parent.parent.parent.parent.parent / "inc/api/compute/matmul.h").read_text()
    assert "if (out == 'W' && ((llk_pack_mm_no_w[family][mask_group][state][row] >> bit) & 1) != 0) { out = 'B'; } if (out == 'B' && ((llk_pack_mm_no_block[family][mask_group][state][row] >> bit) & 1) != 0) { out = 'T'; }" in src
    assert "if (state == 2 && out != 'W' && ((llk_pack_mm_take_w[family][group - 1][row] >> bit) & 1) != 0) { out = 'W'; }" in src
    assert "if (state == 2 && out == 'T' && ((llk_pack_mm_take_b[family][group - 1][row] >> bit) & 1) != 0) { out = 'B'; }" in src
    assert "if (pack_tile_num_faces[i] != 4 || rows != FACE_R_DIM) { if (rows == FACE_R_DIM) { return pack_tile_num_faces[i] == 1 ? 0 : 1; } return rows >= 8 ? 2 : (rows >= 4 ? 3 : 4); }" in src
    assert "if (fmt == static_cast<std::uint32_t>(DataFormat::Float16_b)) { return 10; } if (fmt == static_cast<std::uint32_t>(DataFormat::Float32)) { return 11; } return datum_size_in_bytes(fmt) == 4 ? 8 : 9;" in src
    assert "return llk_pack_class_tab.row[output_id];" in src
    assert "llk_pack_block_form<is_fp32_dest_acc_en, matmul_kt>(output_id, ntiles, first_block ? (pack_reconfigured ? 1 : 0) : 2, pack_ctx[0] != 0);" in src
    assert "llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_init_form, llk_pack_init_form_dc), llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_reconf_form, llk_pack_reconf_form_dc), llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_next_form, llk_pack_next_form_dc)};" in src
    assert "llk_pack_block_form_of<is_fp32_dest_acc_en, matmul_kt>(row, n, tables[state], state, ctx0 + ctx);" in src
    assert "return llk_pack_src32_family<is_fp32_dest_acc_en> && llk_pack_unpack_to_dest ? u : dc;" in src
    assert "#if defined(LLK_UNPACK_TO_DEST_ANY) inline constexpr bool llk_pack_unpack_to_dest = LLK_UNPACK_TO_DEST_ANY != 0; #else inline constexpr bool llk_pack_unpack_to_dest = true; #endif" in src
    assert "closed_pays" not in src and "closed_loses" not in src and "LLK_PACK_BLOCK_MIN_TILES" not in src


def _alt_letter(family, cls, col, count):
    fmt = {"bfp8": "Bfp8_b", "f32": "Float32", "f16b": "Float16_b"}[cls]
    form = pack_block_form(count + 1, fmt, "reconf", fp32_dest=col % 2 == 1, full_sync=col >= 2, src32=bool(family), u2d=False, after_matmul=False)
    return {"per_tile": "T", "block": "B", "w_offset": "W"}[form]


def test_alt_kernel_mirrors_reconf_rows():
    """pack_block_alt_test.cpp holds the reconfigure forms of its three formats (Bfp8_b, Float32, Float16_b) as llk_pack_block takes them in a LoFi kernel without a matmul init or an unpack to DEST, with and without a 32 bit CB."""
    src = ALT.read_text()
    m = re.search(r"constexpr char alt_reconf_form\[2\]\[3\]\[4\]\[17\] = \{(.*?)\n\};", src, re.S)
    assert m
    rows = [re.findall(r"\"([TBW]{16})\"", r) for r in re.findall(r"\{((?:\"[TBW]{16}\",? ?){4})\}", m.group(1))]
    expect = [["".join(_alt_letter(f, c, col, n) for n in range(16)) for col in range(4)] for f in (0, 1) for c in ("bfp8", "f32", "f16b")]
    assert rows == expect
