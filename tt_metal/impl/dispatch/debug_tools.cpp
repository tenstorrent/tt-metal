#include "debug_tools.hpp"

namespace internal {

void match_device_program_data_with_host_program_data(const char* host_file, const char* device_file) {



    std::ifstream host_dispatch_dump_file;
    std::ifstream device_dispatch_dump_file;

    host_dispatch_dump_file.open(host_file);
    device_dispatch_dump_file.open(device_file);

    vector<pair<string, vector<string>>> host_map;


    string line;
    string type;

    while (std::getline(host_dispatch_dump_file, line)) {

        if (line.find("*") != string::npos) {
            continue;
        } else if (line.find("BINARY SPAN") != string::npos or line.find("SEM") != string::npos or line.find("CB") != string::npos) {
            type = line;
        } else {
            vector<string> host_data = {line};
            while (std::getline(host_dispatch_dump_file, line) and (line.find("*") == string::npos)) {
                host_data.push_back(line);
            }
            host_map.push_back(make_pair(type, std::move(host_data)));
        }
    }

    vector<vector<string>> device_map;
    vector<string> device_data;
    while (std::getline(device_dispatch_dump_file, line) and line != "EXIT_CONDITION") {
        if (line == "CHUNK") {
            if (not device_data.empty()) {
                device_map.push_back(device_data);
            }
            device_data.clear();
        } else {
            device_data.push_back(line);
        }
    }
    std::getline(device_dispatch_dump_file, line);
    device_map.push_back(device_data);

    bool all_match = true;
    for (const auto& [type, host_data] : host_map) {
        bool match = false;

        for (const vector<string>& device_data : device_map) {
            if (host_data == device_data) {
                tt::log_info("Matched on {}", type);
                match = true;
                break;
            }
        }

        if (not match) {
            tt::log_info("Mismatch between host and device program data on {}", type);
        }
        all_match &= match;
    }

    host_dispatch_dump_file.close();
    device_dispatch_dump_file.close();

    if (all_match) {
        tt::log_info("Full match between host and device program data");
    }
}

void wait_for_program_vector_to_arrive_and_compare_to_host_program_vector(
    const char* DISPATCH_MAP_DUMP, Device* device) {
    std::string device_dispatch_dump_file_name = "device_" + std::string(DISPATCH_MAP_DUMP);
    while (true) {
        std::ifstream device_dispatch_dump_file;
        device_dispatch_dump_file.open(device_dispatch_dump_file_name);
        std::string line;
        while (!device_dispatch_dump_file.eof()) {
            std::getline(device_dispatch_dump_file, line);

            if (line.find("EXIT_CONDITION") != string::npos) {
                device_dispatch_dump_file.close();

                match_device_program_data_with_host_program_data(
                    DISPATCH_MAP_DUMP, device_dispatch_dump_file_name.c_str());
                CloseDevice(device);
                exit(0);
            }
        }
    }
}

}  // end namespace internal
