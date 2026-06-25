// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <iostream>

/**
 * LUTLoader - Utility class for loading lookup tables from binary files
 *
 * File format: [uint32_t size][float32 values...]
 *   - First 4 bytes: Number of entries (uint32_t)
 *   - Remaining bytes: Float32 values in sequence
 */
class LUTLoader {
public:
    /**
     * Load LUT from binary file (variable size, no validation)
     *
     * @param filename Path to .lut file
     * @return Vector of float values
     * @throws std::runtime_error if file cannot be opened or read
     */
    static std::vector<float> load_from_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open LUT file: " + filename);
        }

        // Read size from header
        uint32_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));

        if (!file || size == 0 || size > 1000000) {  // Sanity check
            throw std::runtime_error(
                "Invalid LUT file header in: " + filename + " (size=" + std::to_string(size) + ")");
        }

        // Read float values
        std::vector<float> lut_data(size);
        file.read(reinterpret_cast<char*>(lut_data.data()), size * sizeof(float));

        if (!file) {
            throw std::runtime_error(
                "Failed to read LUT data from: " + filename + " (expected " + std::to_string(size) + " floats)");
        }

        file.close();
        return lut_data;
    }

    /**
     * Load LUT and validate size matches expected
     *
     * @tparam ExpectedSize Expected number of LUT entries
     * @param filename Path to .lut file
     * @return Vector of float values
     * @throws std::runtime_error if size doesn't match
     */
    template <size_t ExpectedSize>
    static std::vector<float> load_and_validate(const std::string& filename) {
        auto lut_data = load_from_file(filename);

        if (lut_data.size() != ExpectedSize) {
            throw std::runtime_error(
                "LUT size mismatch in " + filename + ": " + "file has " + std::to_string(lut_data.size()) +
                " entries, expected " + std::to_string(ExpectedSize) +
                "\nRegenerate LUTs with correct size or recompile with matching LUT_SIZE");
        }

        return lut_data;
    }

    /**
     * Save LUT data as human-readable text file (for debugging)
     *
     * @param lut_data Vector of LUT values
     * @param filename Output text file path
     */
    static void save_as_text(const std::vector<float>& lut_data, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to create text file: " + filename);
        }

        file << "# LUT Data - " << lut_data.size() << " entries\n";
        file << "# Index\tValue\n";

        for (size_t i = 0; i < lut_data.size(); i++) {
            file << i << "\t" << lut_data[i] << "\n";
        }

        file.close();
        std::cout << "Saved LUT as text to: " << filename << std::endl;
    }

    /**
     * Print summary statistics about LUT
     */
    static void print_stats(const std::vector<float>& lut_data) {
        if (lut_data.empty()) {
            std::cout << "LUT is empty" << std::endl;
            return;
        }

        float min_val = lut_data[0];
        float max_val = lut_data[0];
        double sum = 0.0;

        for (float val : lut_data) {
            if (val < min_val) {
                min_val = val;
            }
            if (val > max_val) {
                max_val = val;
            }
            sum += val;
        }

        double mean = sum / lut_data.size();

        std::cout << "LUT Statistics:" << std::endl;
        std::cout << "  Size: " << lut_data.size() << std::endl;
        std::cout << "  Min: " << min_val << std::endl;
        std::cout << "  Max: " << max_val << std::endl;
        std::cout << "  Mean: " << mean << std::endl;
    }
};
