// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#ifdef TTSIM_ENABLED
class TTSIM_Dump_Guard {
private:
    bool dump_dst;

public:
    TTSIM_Dump_Guard(bool dump_dst = false) {
        this->dump_dst = dump_dst;
        if (this->dump_dst) {
            __asm__ volatile(".word 0x4000007F");
        } else {
            __asm__ volatile(".word 0x0000007F");
        }
    }
    ~TTSIM_Dump_Guard() {
        if (this->dump_dst) {
            __asm__ volatile(".word 0xC000007F");
        } else {
            __asm__ volatile(".word 0x8000007F");
        }
    }
};
#define TTSIM_DUMP_DST true
#define TTSIM_TENSIX_DUMP(dump_dst) auto dump_guard = TTSIM_Dump_Guard(dump_dst)
#define TTSIM_START_TENSIX_DUMP(dump_dst) auto dump_guard_ptr = new TTSIM_Dump_Guard(dump_dst)
#define TTSIM_END_TENSIX_DUMP delete dump_guard_ptr
#else
#define TTSIM_TENSIX_DUMP(dump_dst)
#define TTSIM_START_TENSIX_DUMP(dump_dst)
#define TTSIM_END_TENSIX_DUMP
#define TTSIM_DUMP_DST
#endif
