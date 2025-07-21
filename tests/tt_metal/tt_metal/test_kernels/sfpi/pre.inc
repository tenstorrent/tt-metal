#define FAIL_IF(expr)                                                  \
    v_if(failed == 0 && (expr)) { failed = vUInt(0x4000 + __LINE__); } \
    v_endif

    vUInt failed = vUInt(0);
