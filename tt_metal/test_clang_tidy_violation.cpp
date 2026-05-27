// NOLINT-FREE TEST FILE — intentional clang-tidy violations for CI gate testing
// This file should be caught by code-analysis and blocked by pr-gate.
// See: brain/code-analysis-to-pr-gate

#include <cstddef>

// violation: modernize-use-nullptr
// NULL should be nullptr in C++11 and later
void test_null_violation() {
    int* ptr = NULL;  // NOLINT-purposely-absent
    (void)ptr;
}

// violation: modernize-use-override
// Derived class method should use 'override' keyword
class Base {
public:
    virtual void foo() {}
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    void foo() {}  // missing 'override'
};
