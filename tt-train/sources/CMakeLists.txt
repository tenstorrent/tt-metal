# ttml must come first: examples reuse the ttml_pch precompiled header, and
# PRECOMPILE_HEADERS_REUSE_FROM requires the target to already exist.
add_subdirectory(ttml)
add_subdirectory(examples)
