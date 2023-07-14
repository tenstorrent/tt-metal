 #ifdef TRISC_MATH
 #define MATH(x) x
 #define MAIN math_main()
 #else
 #define MATH(x)
 #endif

 #ifdef TRISC_PACK
 #define PACK(x) x
 #define MAIN pack_main()
 #else
 #define PACK(x)
 #endif

 #ifdef TRISC_UNPACK
 #define UNPACK(x) x
 #define MAIN unpack_main()
 #else
 #define UNPACK(x)
 #endif


namespace NAMESPACE {
void MAIN {
}
}
