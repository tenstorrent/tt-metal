// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// #include <string.h>
#include <cstring>
#ifdef TTSIM_ENABLED

inline void issue_character_instruction(char c) {
    switch (c) {
        case 0: __asm__ volatile(".word 0x0000007F"); break;
        case 1: __asm__ volatile(".word 0x0000017F"); break;
        case 2: __asm__ volatile(".word 0x0000027F"); break;
        case 3: __asm__ volatile(".word 0x0000037F"); break;
        case 4: __asm__ volatile(".word 0x0000047F"); break;
        case 5: __asm__ volatile(".word 0x0000057F"); break;
        case 6: __asm__ volatile(".word 0x0000067F"); break;
        case 7: __asm__ volatile(".word 0x0000077F"); break;
        case 8: __asm__ volatile(".word 0x0000087F"); break;
        case 9: __asm__ volatile(".word 0x0000097F"); break;
        case 10: __asm__ volatile(".word 0x00000A7F"); break;
        case 11: __asm__ volatile(".word 0x00000B7F"); break;
        case 12: __asm__ volatile(".word 0x00000C7F"); break;
        case 13: __asm__ volatile(".word 0x00000D7F"); break;
        case 14: __asm__ volatile(".word 0x00000E7F"); break;
        case 15: __asm__ volatile(".word 0x00000F7F"); break;
        case 16: __asm__ volatile(".word 0x0000107F"); break;
        case 17: __asm__ volatile(".word 0x0000117F"); break;
        case 18: __asm__ volatile(".word 0x0000127F"); break;
        case 19: __asm__ volatile(".word 0x0000137F"); break;
        case 20: __asm__ volatile(".word 0x0000147F"); break;
        case 21: __asm__ volatile(".word 0x0000157F"); break;
        case 22: __asm__ volatile(".word 0x0000167F"); break;
        case 23: __asm__ volatile(".word 0x0000177F"); break;
        case 24: __asm__ volatile(".word 0x0000187F"); break;
        case 25: __asm__ volatile(".word 0x0000197F"); break;
        case 26: __asm__ volatile(".word 0x00001A7F"); break;
        case 27: __asm__ volatile(".word 0x00001B7F"); break;
        case 28: __asm__ volatile(".word 0x00001C7F"); break;
        case 29: __asm__ volatile(".word 0x00001D7F"); break;
        case 30: __asm__ volatile(".word 0x00001E7F"); break;
        case 31: __asm__ volatile(".word 0x00001F7F"); break;
        case 32: __asm__ volatile(".word 0x0000207F"); break;
        case 33: __asm__ volatile(".word 0x0000217F"); break;
        case 34: __asm__ volatile(".word 0x0000227F"); break;
        case 35: __asm__ volatile(".word 0x0000237F"); break;
        case 36: __asm__ volatile(".word 0x0000247F"); break;
        case 37: __asm__ volatile(".word 0x0000257F"); break;
        case 38: __asm__ volatile(".word 0x0000267F"); break;
        case 39: __asm__ volatile(".word 0x0000277F"); break;
        case 40: __asm__ volatile(".word 0x0000287F"); break;
        case 41: __asm__ volatile(".word 0x0000297F"); break;
        case 42: __asm__ volatile(".word 0x00002A7F"); break;
        case 43: __asm__ volatile(".word 0x00002B7F"); break;
        case 44: __asm__ volatile(".word 0x00002C7F"); break;
        case 45: __asm__ volatile(".word 0x00002D7F"); break;
        case 46: __asm__ volatile(".word 0x00002E7F"); break;
        case 47: __asm__ volatile(".word 0x00002F7F"); break;
        case 48: __asm__ volatile(".word 0x0000307F"); break;
        case 49: __asm__ volatile(".word 0x0000317F"); break;
        case 50: __asm__ volatile(".word 0x0000327F"); break;
        case 51: __asm__ volatile(".word 0x0000337F"); break;
        case 52: __asm__ volatile(".word 0x0000347F"); break;
        case 53: __asm__ volatile(".word 0x0000357F"); break;
        case 54: __asm__ volatile(".word 0x0000367F"); break;
        case 55: __asm__ volatile(".word 0x0000377F"); break;
        case 56: __asm__ volatile(".word 0x0000387F"); break;
        case 57: __asm__ volatile(".word 0x0000397F"); break;
        case 58: __asm__ volatile(".word 0x00003A7F"); break;
        case 59: __asm__ volatile(".word 0x00003B7F"); break;
        case 60: __asm__ volatile(".word 0x00003C7F"); break;
        case 61: __asm__ volatile(".word 0x00003D7F"); break;
        case 62: __asm__ volatile(".word 0x00003E7F"); break;
        case 63: __asm__ volatile(".word 0x00003F7F"); break;
        case 64: __asm__ volatile(".word 0x0000407F"); break;
        case 65: __asm__ volatile(".word 0x0000417F"); break;
        case 66: __asm__ volatile(".word 0x0000427F"); break;
        case 67: __asm__ volatile(".word 0x0000437F"); break;
        case 68: __asm__ volatile(".word 0x0000447F"); break;
        case 69: __asm__ volatile(".word 0x0000457F"); break;
        case 70: __asm__ volatile(".word 0x0000467F"); break;
        case 71: __asm__ volatile(".word 0x0000477F"); break;
        case 72: __asm__ volatile(".word 0x0000487F"); break;
        case 73: __asm__ volatile(".word 0x0000497F"); break;
        case 74: __asm__ volatile(".word 0x00004A7F"); break;
        case 75: __asm__ volatile(".word 0x00004B7F"); break;
        case 76: __asm__ volatile(".word 0x00004C7F"); break;
        case 77: __asm__ volatile(".word 0x00004D7F"); break;
        case 78: __asm__ volatile(".word 0x00004E7F"); break;
        case 79: __asm__ volatile(".word 0x00004F7F"); break;
        case 80: __asm__ volatile(".word 0x0000507F"); break;
        case 81: __asm__ volatile(".word 0x0000517F"); break;
        case 82: __asm__ volatile(".word 0x0000527F"); break;
        case 83: __asm__ volatile(".word 0x0000537F"); break;
        case 84: __asm__ volatile(".word 0x0000547F"); break;
        case 85: __asm__ volatile(".word 0x0000557F"); break;
        case 86: __asm__ volatile(".word 0x0000567F"); break;
        case 87: __asm__ volatile(".word 0x0000577F"); break;
        case 88: __asm__ volatile(".word 0x0000587F"); break;
        case 89: __asm__ volatile(".word 0x0000597F"); break;
        case 90: __asm__ volatile(".word 0x00005A7F"); break;
        case 91: __asm__ volatile(".word 0x00005B7F"); break;
        case 92: __asm__ volatile(".word 0x00005C7F"); break;
        case 93: __asm__ volatile(".word 0x00005D7F"); break;
        case 94: __asm__ volatile(".word 0x00005E7F"); break;
        case 95: __asm__ volatile(".word 0x00005F7F"); break;
        case 96: __asm__ volatile(".word 0x0000607F"); break;
        case 97: __asm__ volatile(".word 0x0000617F"); break;
        case 98: __asm__ volatile(".word 0x0000627F"); break;
        case 99: __asm__ volatile(".word 0x0000637F"); break;
        case 100: __asm__ volatile(".word 0x0000647F"); break;
        case 101: __asm__ volatile(".word 0x0000657F"); break;
        case 102: __asm__ volatile(".word 0x0000667F"); break;
        case 103: __asm__ volatile(".word 0x0000677F"); break;
        case 104: __asm__ volatile(".word 0x0000687F"); break;
        case 105: __asm__ volatile(".word 0x0000697F"); break;
        case 106: __asm__ volatile(".word 0x00006A7F"); break;
        case 107: __asm__ volatile(".word 0x00006B7F"); break;
        case 108: __asm__ volatile(".word 0x00006C7F"); break;
        case 109: __asm__ volatile(".word 0x00006D7F"); break;
        case 110: __asm__ volatile(".word 0x00006E7F"); break;
        case 111: __asm__ volatile(".word 0x00006F7F"); break;
        case 112: __asm__ volatile(".word 0x0000707F"); break;
        case 113: __asm__ volatile(".word 0x0000717F"); break;
        case 114: __asm__ volatile(".word 0x0000727F"); break;
        case 115: __asm__ volatile(".word 0x0000737F"); break;
        case 116: __asm__ volatile(".word 0x0000747F"); break;
        case 117: __asm__ volatile(".word 0x0000757F"); break;
        case 118: __asm__ volatile(".word 0x0000767F"); break;
        case 119: __asm__ volatile(".word 0x0000777F"); break;
        case 120: __asm__ volatile(".word 0x0000787F"); break;
        case 121: __asm__ volatile(".word 0x0000797F"); break;
        case 122: __asm__ volatile(".word 0x00007A7F"); break;
        case 123: __asm__ volatile(".word 0x00007B7F"); break;
        case 124: __asm__ volatile(".word 0x00007C7F"); break;
        case 125: __asm__ volatile(".word 0x00007D7F"); break;
        case 126: __asm__ volatile(".word 0x00007E7F"); break;
        case 127: __asm__ volatile(".word 0x00007F7F"); break;
        default: break;
    }
}
class TTSIM_Dump_Guard {
private:
    bool dump_dst;
    const char* title;

public:
    TTSIM_Dump_Guard(const char* title, bool dump_dst) {
        this->dump_dst = dump_dst;
        this->title = title;
        if (this->title != nullptr && strlen(this->title) > 0) {
            for (int i = 0, n = strlen(this->title); i < n; i++) {
                issue_character_instruction(this->title[i]);
            }
        }
        if (this->dump_dst) {
            __asm__ volatile(".word 0x400000FF");
        } else {
            __asm__ volatile(".word 0x000000FF");
        }
    }
    ~TTSIM_Dump_Guard() {
        if (this->title != nullptr && strlen(this->title) > 0) {
            for (int i = 0, n = strlen(this->title); i < n; i++) {
                issue_character_instruction(this->title[i]);
            }
        }
        if (this->dump_dst) {
            __asm__ volatile(".word 0xC00000FF");
        } else {
            __asm__ volatile(".word 0x800000FF");
        }
    }
};
#define TTSIM_DUMP_DST true
#define TTSIM_TENSIX_DUMP(title, dump_dst) auto dump_guard = TTSIM_Dump_Guard(title, dump_dst)
#define TTSIM_START_TENSIX_DUMP(title, dump_dst) auto dump_guard_ptr = new TTSIM_Dump_Guard(title, dump_dst)
#define TTSIM_END_TENSIX_DUMP delete dump_guard_ptr
#else
#define TTSIM_TENSIX_DUMP(title, dump_dst)
#define TTSIM_START_TENSIX_DUMP(title, dump_dst)
#define TTSIM_END_TENSIX_DUMP
#define TTSIM_DUMP_DST
#endif
