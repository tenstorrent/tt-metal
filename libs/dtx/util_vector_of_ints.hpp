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

using namespace std;

// ========================================================
//                      PRINT / DEBUG
// ========================================================
string v2s(vector<int> vector);
string s(int);
string sp(int);

// ========================================================
//                      VECTOR MATH AND MANIPULATION
// ========================================================
vector<int> vector_subtraction(vector<int> a, vector<int> b);
vector<int> vector_addition(vector<int> a, vector<int> b, int const_value = 0);
vector<int> vector_division(vector<int> a, vector<int> b);
vector<int> vector_multiplication(vector<int> a, vector<int> b);
int vector_product(vector<int> a);
vector<int> copy_vector_of_ints(vector<int> a);
vector<int> zeros(int rank);
vector<int> ones(int rank);
vector<int> vector_pad_on_left(vector<int> a, int pad_size, int pad_value);


vector<int> increment(vector<int> current, vector<int> start, vector<int> end);
bool equal(vector<int> a, vector<int> b);
