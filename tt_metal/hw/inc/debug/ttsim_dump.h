#pragma once

class TTSIM_Dump_Guard {
public:
    TTSIM_Dump_Guard() { __asm__ volatile(".word 0x00000000"); }
    ~TTSIM_Dump_Guard() { __asm__ volatile(".word 0x00000001"); }
};

#ifdef ENABLE_TTSIM_TENSIX_DUMP
#define TTSIM_TENSIX_DUMP auto dump_guard = TTSIM_Dump_Guard()
#define TTSIM_START_TENSIX_DUMP auto dump_guard_ptr = new TTSIM_Dump_Guard()
#define TTSIM_END_TENSIX_DUMP delete dump_guard_ptr
#else
#define TTSIM_TENSIX_DUMP
#define TTSIM_START_TENSIX_DUMP
#define TTSIM_END_TENSIX_DUMP
#endif
