#include "inspector.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "program.hpp"
#include <tt-metalium/logger.hpp>
#include <chrono>
#include <iomanip>
#include <ctime>

#define TT_INSPECTOR_THROW(...) \
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_inspector_initialization_is_important()) { \
        TT_THROW(__VA_ARGS__); \
    } else { \
        tt::log_warning(tt::LogInspector, __VA_ARGS__); \
        return; \
    }

#define TT_INSPECTOR_LOG(...) \
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_inspector_warn_on_write_exceptions()) { \
        tt::log_warning(tt::LogInspector, __VA_ARGS__); \
    }

namespace tt::tt_metal {

inspector::Logger::Logger(const std::filesystem::path& logging_path)
    : programs_ostream(logging_path / "programs_log.yaml", std::ios::trunc)
    , kernels_ostream(logging_path / "kernels.yaml", std::ios::trunc) {
    {
        std::ofstream inspector_startup_ostream(logging_path / "startup.yaml", std::ios::trunc);

        if (!inspector_startup_ostream.is_open()) {
            TT_INSPECTOR_THROW("Failed to create inspector file: {}", (logging_path / "startup.yaml").string());
        }
        else {
            // Log current system time and high_resolution_clock time_point
            auto now_system = std::chrono::system_clock::now();
            auto now_highres = std::chrono::high_resolution_clock::now();
            std::time_t now_c = std::chrono::system_clock::to_time_t(now_system);
            auto now_system_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now_system.time_since_epoch()).count();
            auto now_highres_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now_highres.time_since_epoch()).count();

            start_time = now_highres;
            inspector_startup_ostream << "startup_time:\n";
            inspector_startup_ostream << "  system_clock_iso: '" << std::put_time(std::gmtime(&now_c), "%Y-%m-%dT%H:%M:%SZ") << "'\n";
            inspector_startup_ostream << "  system_clock_ns: " << now_system_ns << "\n";
            inspector_startup_ostream << "  high_resolution_clock_ns: " << now_highres_ns << "\n";
        }
    }
    if (!programs_ostream.is_open()) {
        TT_INSPECTOR_THROW("Failed to create inspector file: {}", (logging_path / "programs_log.yaml").string());
    }
    if (!kernels_ostream.is_open()) {
        TT_INSPECTOR_THROW("Failed to create inspector file: {}", (logging_path / "kernels.yaml").string());
    }
}

void inspector::Logger::log_program_created(const inspector::ProgramData& program_data) noexcept {
    try {
        programs_ostream << "- program_created:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(std::chrono::high_resolution_clock::now()) << "\n";
        programs_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program created: {}", e.what());
    }
}

void inspector::Logger::log_program_destroyed(const inspector::ProgramData& program_data) noexcept {
    try {
        programs_ostream << "- program_destroyed:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(std::chrono::high_resolution_clock::now()) << "\n";
        programs_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program destroyed: {}", e.what());
    }
}

void inspector::Logger::log_program_compile_started(const inspector::ProgramData& program_data) noexcept {
    try {
        programs_ostream << "- program_compile_started:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(program_data.compile_started_timestamp) << "\n";
        programs_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program compile started: {}", e.what());
    }
}

void inspector::Logger::log_program_compile_already_exists(const inspector::ProgramData& program_data) noexcept {
    try {
        programs_ostream << "- program_compile_already_exists:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(program_data.compile_started_timestamp) << "\n";
        programs_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program compile already exists: {}", e.what());
    }
}

void inspector::Logger::log_program_kernel_compile_finished(const inspector::ProgramData& program_data, const inspector::KernelData& kernel_data) noexcept {
    try {
        auto timestamp = std::chrono::high_resolution_clock::now();
        programs_ostream << "- program_kernel_compile_finished:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(timestamp) << "\n";
        programs_ostream << "    duration_ns: " << std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp - program_data.compile_started_timestamp).count() << "\n";
        programs_ostream << "    watcher_kernel_id: " << kernel_data.watcher_kernel_id << "\n";
        programs_ostream << "    name: " << kernel_data.name << "\n";
        programs_ostream << "    path: " << kernel_data.path << "\n";
        programs_ostream << "    source: " << kernel_data.source << "\n";
        programs_ostream.flush();
        kernels_ostream << "- kernel:\n";
        kernels_ostream << "    watcher_kernel_id: " << kernel_data.watcher_kernel_id << "\n";
        kernels_ostream << "    name: " << kernel_data.name << "\n";
        kernels_ostream << "    path: " << kernel_data.path << "\n";
        kernels_ostream << "    source: " << kernel_data.source << "\n";
        kernels_ostream << "    program_id: " << program_data.program_id << "\n";
        kernels_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program kernel compile finished: {}", e.what());
    }
}

void inspector::Logger::log_program_compile_finished(const inspector::ProgramData& program_data) noexcept {
    try {
        programs_ostream << "- program_compile_finished:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(program_data.compile_finished_timestamp) << "\n";
        programs_ostream << "    duration_ns: " << std::chrono::duration_cast<std::chrono::nanoseconds>(program_data.compile_finished_timestamp - program_data.compile_started_timestamp).count() << "\n";
        programs_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program compile finished: {}", e.what());
    }
}

void inspector::Logger::log_program_binary_status_change(const inspector::ProgramData& program_data, std::size_t device_id, ProgramBinaryStatus status) noexcept {
    try {
        programs_ostream << "- program_binary_status_change:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(std::chrono::high_resolution_clock::now()) << "\n";
        programs_ostream << "    device_id: " << device_id << "\n";
        programs_ostream << "    status: ";
        switch (status) {
            case ProgramBinaryStatus::NotSent:
                programs_ostream << "NotSent" << "\n";
                break;
            case ProgramBinaryStatus::InFlight:
                programs_ostream << "InFlight" << "\n";
                break;
            case ProgramBinaryStatus::Committed:
                programs_ostream << "Committed" << "\n";
                break;
        }
        programs_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program enqueued: {}", e.what());
    }
}

// TODO: Initialization must be called explicitly by the user so we can fail startup if we cannot create log files...
Inspector::Inspector()
    : logger(MetalContext::instance().rtoptions().get_inspector_log_path()) {
}

bool Inspector::is_enabled() {
    return tt::tt_metal::MetalContext::instance().rtoptions().get_inspector_enabled();
}

void Inspector::program_created(
    const detail::ProgramImpl* program) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto& instance = Inspector::instance();
        std::lock_guard<std::mutex> lock(instance.programs_mutex);
        auto& program_data = instance.programs_data[program->get_id()];
        program_data.program=program->weak_from_this();
        program_data.program_id=program->get_id();
        instance.logger.log_program_created(program_data);
    }
    catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program created: {}", e.what());
    }
}

void Inspector::program_destroyed(
    const detail::ProgramImpl* program) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto& instance = Inspector::instance();
        std::lock_guard<std::mutex> lock(instance.programs_mutex);
        auto& program_data = instance.programs_data[program->get_id()];
        instance.logger.log_program_destroyed(program_data);
        instance.programs_data.erase(program->get_id());
    }
    catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program destroyed: {}", e.what());
    }
}

void Inspector::program_compile_started(
    const detail::ProgramImpl* program,
    const IDevice* device,
    uint32_t build_key) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto& instance = Inspector::instance();
        std::lock_guard<std::mutex> lock(instance.programs_mutex);
        auto& program_data = instance.programs_data[program->get_id()];
        program_data.compile_started_timestamp = std::chrono::high_resolution_clock::now();
        instance.logger.log_program_compile_started(program_data);
    }
    catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program destroyed: {}", e.what());
    }
}

void Inspector::program_compile_already_exists(
    const detail::ProgramImpl* program,
    const IDevice* device,
    uint32_t build_key) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto& instance = Inspector::instance();
        std::lock_guard<std::mutex> lock(instance.programs_mutex);
        auto& program_data = instance.programs_data[program->get_id()];
        instance.logger.log_program_compile_already_exists(program_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program compile already exists: {}", e.what());
    }
}

void Inspector::program_kernel_compile_finished(
    const detail::ProgramImpl* program,
    const IDevice* device,
    const std::shared_ptr<Kernel>& kernel,
    const tt::tt_metal::JitBuildOptions& build_options) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto& instance = Inspector::instance();
        std::lock_guard<std::mutex> lock(instance.programs_mutex);
        auto& program_data = instance.programs_data[program->get_id()];
        auto& kernel_data = program_data.kernels[kernel->get_watcher_kernel_id()];
        kernel_data.kernel = kernel;
        kernel_data.watcher_kernel_id = kernel->get_watcher_kernel_id();
        kernel_data.name = kernel->name();
        kernel_data.path = build_options.path;
        kernel_data.source = kernel->kernel_source().source_;
        instance.logger.log_program_kernel_compile_finished(program_data, kernel_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program kernel compile finished: {}", e.what());
    }
}

void Inspector::program_compile_finished(
    const detail::ProgramImpl* program,
    const IDevice* device,
    uint32_t build_key) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto& instance = Inspector::instance();
        std::lock_guard<std::mutex> lock(instance.programs_mutex);
        auto& program_data = instance.programs_data[program->get_id()];
        program_data.compile_finished_timestamp = std::chrono::high_resolution_clock::now();
        instance.logger.log_program_compile_finished(program_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program compile finished: {}", e.what());
    }
}

void Inspector::program_set_binary_status(
    const detail::ProgramImpl* program,
    std::size_t device_id,
    ProgramBinaryStatus status) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto& instance = Inspector::instance();
        std::lock_guard<std::mutex> lock(instance.programs_mutex);
        auto& program_data = instance.programs_data[program->get_id()];
        program_data.binary_status_per_device[device_id] = status;
        instance.logger.log_program_binary_status_change(program_data, device_id, status);
    }
    catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program binary status change: {}", e.what());
    }
}

}  // namespace tt::tt_metal
