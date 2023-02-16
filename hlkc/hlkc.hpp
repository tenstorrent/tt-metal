#pragma once

#include <string>

////////////////////////////////////////////////////////
// If any of the following is defined macros is defined
// rose.h will undefine it and print warning message.
# undef DEBUG
# undef TRACE
# undef WHERE
# undef MARCH
# undef INFO
# undef WARN
# undef ERROR
# undef FATAL

#include "rose.h"

using std::string;

enum LLKTarget {
    UNPACK = 0,
    MATH = 1,
    PACK = 2,
    STRUCT_INIT_GEN = 3
};

enum DeviceEnum {
    GRAYSKULL = 0,
    WORMHOLE = 1,
    WORMHOLE_B0 = 2,
    BLACKHOLE = 3,
    INVALID_DEVICE = 3
};

LLKTarget target_enum_from_string(string target_str);
string target_string_from_enum(LLKTarget target_enum);
