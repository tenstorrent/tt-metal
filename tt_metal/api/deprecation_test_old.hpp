// Test file with OLD deprecations (will be 40+ days old)
#pragma once

namespace tt_metal {

// REMOVED: Old deprecation (was 40+ days old - SAFE)
void oldDeprecatedFunc1();

// REMOVED: Old deprecation (was 40+ days old - SAFE)
int oldDeprecatedFunc2(int x);

// Old deprecated class
[[deprecated("This class is obsolete")]]
class OldDeprecatedClass {
public:
    void method();
};

} // namespace tt_metal