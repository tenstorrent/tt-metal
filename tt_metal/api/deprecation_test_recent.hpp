// Test file with RECENT deprecations (will be 10 days old)
#pragma once

namespace tt_metal {

// Recent deprecation - NOT safe to remove yet (only 10 days old)
[[deprecated("Use newRecentFunc1 instead")]]
void recentDeprecatedFunc1();

// Another recent deprecation
[[deprecated]]
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