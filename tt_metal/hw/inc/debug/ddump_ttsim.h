#pragma once

class TTSIM_Dump_Guard {
public:
    TTSIM_Dump_Guard() { __asm__ volatile(".word 0x00000000"); }
    ~TTSIM_Dump_Guard() { __asm__ volatile(".word 0x00000001"); }
};

#define TTSIM_DDUMP auto dump_guard = TTSIM_Dump_Guard()
#define TTSIM_START_DDUMP auto dump_guard_ptr = new TTSIM_Dump_Guard()
#define TTSIM_END_DDUMP delete dump_guard_ptr
