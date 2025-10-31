#pragma once
#ifdef USING_TTSIM
class TTSIM_Dump_Guard {
private:
    bool dump_dst;

public:
    TTSIM_Dump_Guard(bool dump_dst = false) {
        this->dump_dst = dump_dst;
        if (this->dump_dst) {
            __asm__ volatile(".word 0x4000003F");
        } else {
            __asm__ volatile(".word 0x0000003F");
        }
    }
    ~TTSIM_Dump_Guard() {
        if (this->dump_dst) {
            __asm__ volatile(".word 0xC000003F");
        } else {
            __asm__ volatile(".word 0x8000003F");
        }
    }
};
#define TTSIM_DUMP_DST true
#define TTSIM_TENSIX_DUMP(dump_dst) auto dump_guard = TTSIM_Dump_Guard(dump_dst)
#define TTSIM_START_TENSIX_DUMP auto dump_guard_ptr = new TTSIM_Dump_Guard(dump_dst)
#define TTSIM_END_TENSIX_DUMP delete dump_guard_ptr
#else
#define TTSIM_TENSIX_DUMP
#define TTSIM_START_TENSIX_DUMP
#define TTSIM_END_TENSIX_DUMP
#define TTSIM_DUMP_DST
#endif
