 #ifdef TRISC_MATH
 #define MAIN math_main()
 #endif

 #ifdef TRISC_PACK
 #define MAIN pack_main()
 #endif

 #ifdef TRISC_UNPACK
 #define MAIN unpack_main()
 #endif

namespace NAMESPACE {
void MAIN {
}
}
