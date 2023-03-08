#pragma once
#include <string>

namespace tt {
namespace reportify {

bool initalize_reportify_directory(const std::string& reportify_dir, const std::string& test_name);
std::string get_default_reportify_path(const std::string& test_name);
std::string get_pass_reports_relative_directory();
std::string get_router_report_relative_directory();
std::string get_memory_report_relative_directory();
std::string get_epoch_type_report_relative_directory();
std::string get_epoch_id_report_relative_directory();

} // namespace reportify
} // namespace tt
