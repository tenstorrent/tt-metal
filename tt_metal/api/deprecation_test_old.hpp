// Test file with OLD deprecations (will be 40+ days old)
#pragma once

namespace tt_metal {

// Old deprecation - safe to remove after 30 days
[[deprecated("Use newOldFunc1 instead")]]
void oldDeprecatedFunc1();

// Old deprecation without message
[[deprecated]]
int oldDeprecatedFunc2(int x);

// Old deprecated class
[[deprecated("This class is obsolete")]]
class OldDeprecatedClass {
public:
    void method();
};

} // namespace tt_metal