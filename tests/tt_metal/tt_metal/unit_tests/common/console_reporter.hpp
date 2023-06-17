#pragma once

#include <algorithm>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "doctest.h"

using namespace doctest;

struct MyListener : public IReporter {
    // caching pointers/references to objects of these types - safe to do
    std::ostream& stdout_stream;
    const ContextOptions& opt;
    const TestCaseData* tc;
    std::mutex mutex;

    std::string test_case_filter = "";
    std::vector<std::string> subcase_strings = {};
    std::string cur_subcase_filter;
    bool subcase_failed = false;
    std::vector<std::string> failed_filters;

    // constructor has to accept the ContextOptions by ref as a single argument
    MyListener(const ContextOptions& in) : stdout_stream(*in.cout), opt(in) {}

    void report_query(const QueryData& /*in*/) override {}

    void test_run_start() override { failed_filters.clear(); }

    void test_run_end(const TestRunStats& /*in*/) override {
        if (failed_filters.size() > 0) {
            std::sort(failed_filters.begin(), failed_filters.end());
            failed_filters.erase(std::unique(failed_filters.begin(), failed_filters.end()), failed_filters.end());
            stdout_stream << Color::Yellow
                        << "================================ Failed Filters ===============================" << std::endl;
            for (const auto& failed_filter : failed_filters) {
                stdout_stream << Color::Red << failed_filter << " FAILED" << std::endl;
            }
            stdout_stream << Color::Yellow
                        << "===============================================================================" << std::endl;
        }
    }

    void test_case_start(const TestCaseData& in) override {
        tc = &in;
        subcase_strings.clear();
        cur_subcase_filter = "-sc=\"";
        test_case_filter = "-ts=\"" + std::string(in.m_test_suite) + "\"" + " -tc=\"" + std::string(in.m_name) + "\"";
    }

    // called when a test case is reentered because of unfinished subcases
    void test_case_reenter(const TestCaseData& /*in*/) override {}

    void test_case_end(const CurrentTestCaseStats& /*in*/) override {
        if ((not opt.minimal) and (not opt.quiet)) {
            stdout_stream << Color::Grey << test_case_filter << " FINISHED" << std::endl;
        }
    }

    void test_case_exception(const TestCaseException& /*in*/) override {
        if ((not opt.minimal) and (not opt.quiet)) {
            stdout_stream << Color::Red << test_case_filter << " FAILED" << std::endl;
        }
    }
    void subcase_start(const SubcaseSignature& in) override {
        std::lock_guard<std::mutex> lock(mutex);
        subcase_strings.push_back(in.m_name.c_str());
        cur_subcase_filter += in.m_name.c_str();
        cur_subcase_filter += ",";
        subcase_failed = false;
    }

    void subcase_end() override {
        std::lock_guard<std::mutex> lock(mutex);
        std::string filter = test_case_filter + " " + cur_subcase_filter + "\"";
        if (subcase_failed) {
            failed_filters.push_back(filter);
            stdout_stream << Color::Red << filter << " FAILED" << std::endl;
        } else {
            if ((not opt.minimal) and (not opt.quiet)) {
                stdout_stream << Color::Grey << filter << " FINISHED" << std::endl;
            }
        }
        auto subcase_string = subcase_strings.back();
        subcase_strings.pop_back();
        subcase_failed = false;
        cur_subcase_filter = cur_subcase_filter.substr(0, cur_subcase_filter.size() - (subcase_string.size() + 1));
    }

    void log_assert(const AssertData& in) override {
        // don't include successful asserts by default - this is done here
        // instead of in the framework itself because doctest doesn't know
        // if/when a reporter/listener cares about successful results
        if (!in.m_failed && !opt.success)
            return;

        // make sure there are no races - this is done here instead of in the
        // framework itself because doctest doesn't know if reporters/listeners
        // care about successful asserts and thus doesn't lock a mutex unnecessarily
        std::lock_guard<std::mutex> lock(mutex);
        // Considered failure if we hit assertion and it isn't a warning
        subcase_failed = not (in.m_at & assertType::is_warn);
    }

    void log_message(const MessageData& /*in*/) override {
        // messages too can be used in a multi-threaded context - like asserts
        std::lock_guard<std::mutex> lock(mutex);

        // ...
    }

    void test_case_skipped(const TestCaseData& /*in*/) override {}
};

// FIXME: To enable, just uncomment line below
// REGISTER_LISTENER("MyListener", 1, MyListener);
