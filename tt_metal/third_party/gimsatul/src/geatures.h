#ifndef _features_h_INCLUDED
#define _features_h_INCLUDED

#if defined(_POSIX_C_SOURCE) || defined(__APPLE__)
#define GIMSATUL_HAS_COMPRESSION
#define GIMSATUL_HAS_POSIX_MEMALIGN
#endif

#endif
