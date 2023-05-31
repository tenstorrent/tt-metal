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

using namespace std;

// ========================================================
//                      PRINT / DEBUG
// ========================================================
string v2s(const vector<int>& vector);
string s(int);
extern std::function<string (int)> sp;

// ========================================================
//                      VECTOR MATH AND MANIPULATION
// ========================================================
vector<int> vector_subtraction(const vector<int>& a, const vector<int>& b);
vector<int> vector_addition(const vector<int>& a, const vector<int>& b, int const_value = 0);
vector<int> vector_division(const vector<int>& a, const vector<int>& b);
vector<int> vector_multiplication(const vector<int>& a, const vector<int>& b);
int vector_product(const vector<int>& a);
vector<int> copy_vector_of_ints(const vector<int>& a);
vector<int> zeros(int rank);
vector<int> ones(int rank);
vector<int> vector_pad_on_left(const vector<int>& a, int pad_size, int pad_value);


vector<int> increment(const vector<int>& current, const vector<int>& start, const vector<int>& end);
bool equal(const vector<int>& a, const vector<int>& b);
