// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <functional>

// ========================================================
//                      PRINT / DEBUG
// ========================================================
std::string v2s(const std::vector<int>& vector);
std::string s(int);
extern std::function<std::string (int)> sp;

// ========================================================
//                      VECTOR MATH AND MANIPULATION
// ========================================================
std::vector<int> vector_subtraction(const std::vector<int>& a, const std::vector<int>& b);
std::vector<int> vector_addition(const std::vector<int>& a, const std::vector<int>& b, int const_value = 0);
std::vector<int> vector_division(const std::vector<int>& a, const std::vector<int>& b);
std::vector<int> vector_multiplication(const std::vector<int>& a, const std::vector<int>& b);
int vector_product(const std::vector<int>& a);
std::vector<int> copy_vector_of_ints(const std::vector<int>& a);
std::vector<int> zeros(int rank);
std::vector<int> ones(int rank);
std::vector<int> vector_pad_on_left(const std::vector<int>& a, int pad_size, int pad_value);


std::vector<int> increment(const std::vector<int>& current, const std::vector<int>& start, const std::vector<int>& end);
bool equal(const std::vector<int>& a, const std::vector<int>& b);
