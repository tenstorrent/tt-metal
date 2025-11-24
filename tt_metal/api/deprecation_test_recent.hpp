// Test file with RECENT deprecations (will be 10 days old)
#pragma once

namespace tt_metal {

// VIOLATION: Removed recent deprecation (only 10 days old)
void recentDeprecatedFunc1();

// VIOLATION: Removed recent deprecation (only 10 days old)
struct RecentDeprecatedStruct {
    int value;
};

// Recent deprecated enum
[[deprecated("Use NewEnum instead")]]
enum OldEnum {
    VALUE1,
    VALUE2
};

} // namespace tt_metal