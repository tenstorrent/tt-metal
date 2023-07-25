#ifndef _RISC_ATTRIBS_H_
#define _RISC_ATTRIBS_H_

#define tt_l1_ptr __attribute__((rvtt_l1_ptr))
#define tt_reg_ptr __attribute__((rvtt_reg_ptr))

union tt_uint64_t {
    uint64_t v;
    struct {
        uint32_t hi;
        uint32_t lo;
    };
};

inline __attribute__((always_inline)) uint64_t tt_l1_load(tt_uint64_t tt_l1_ptr *p)
{
    tt_uint64_t v;

    v.hi = p->hi;
    v.lo = p->lo;
    return v.v;
}

#endif // _RISC_ATTRIBS_H_
