#pragma once

#include <bitset>
#include <utility>

#include "src/firmware/riscv/common/fw_debug.h"

/**
 * @brief A hash-based 'map' from a key to it's position in a table of fixed size
 *
 * Not completely dissimilar to a hash set, but it has a different interface.
 * Uses linear probing.
 */
template <
    typename Key,
    std::size_t max_entries_,
    typename KeyStorage = std::array<Key, max_entries_>,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<>>
class StaticHashedKeyTable {
 public:
  using key_type = Key;
  using hasher = Hash;
  using key_equal = KeyEqual;
  using index_type = std::int32_t;
  static constexpr index_type invalid_index = -1;

  struct DoNothingValueMover {
    void operator()(index_type new_index, index_type old_index) { (void)old_index, (void)new_index; }
  };

  StaticHashedKeyTable() : StaticHashedKeyTable(KeyStorage()) {}  // don't want default constructor to be explicit
  explicit StaticHashedKeyTable(KeyStorage key_storage, hasher h = hasher(), key_equal ke = key_equal()) :
      keys(std::move(key_storage)), hasher_instance(std::move(h)), key_equal_instance(std::move(ke)) {
    FWASSERT("can't use nullptr for key storage", keys.data() != nullptr);
  }

  /** @brief Find @p k . @returns (index,was_found) */
  std::pair<index_type, bool> find(const key_type& k) const {
    const auto hash = hasher_instance(k);
    const index_type initial_index = hash % max_entries_;

    bool looped_around = false;
    for (index_type i = initial_index;; ++i) {
      if (std::size_t(i) == max_entries_) {
        i = 0;
        looped_around = true;
      }
      if (i == initial_index && looped_around) {
        break;
      }

      // loop body
      if (not key_is_valid[i]) {
        break;
      }
      if (key_equal_instance(keys[i], k)) {
        return {i, true};
      }
    }
    // did not find
    return {invalid_index, false};
  }

  index_type at(const key_type& k) const {
    const auto [index, valid] = find(k);
    FWASSERT("Could not find entry in a hash table", valid);
    (void)valid;
    return index;
  }

  const key_type& get_key(index_type i) const { return keys[i]; }
  key_type& get_key(index_type i) { return keys[i]; }

  /**
   * @brief Add @p k . Does not overwrite.
   *
   * @return Either (where @p k was inserted, true) or (where @p k was found, false)
   */
  std::pair<index_type, bool> insert(key_type k) {
    const auto hash = hasher_instance(k);
    const auto initial_index = index_type(hash % max_entries_);

    bool looped_around = false;
    for (index_type i = initial_index;; ++i) {
      if (std::size_t(i) == max_entries_) {
        i = 0;
        looped_around = true;
      }
      if (i == initial_index && looped_around) {
        break;
      }

      // loop body
      if (key_is_valid[i]) {
        if (key_equal_instance(keys[i], k)) {
          return {i, false};
        }
      } else {
        keys[i] = std::move(k);
        key_is_valid[i] = true;
        return {i, true};
      }
    }
    FWASSERT("hash table is full", false);  // Or, should there be an separate insert_checked?
    return {invalid_index, false};
  }

  /**
   * @brief find + erase_index
   *
   * See erase_index for further details.
   * @return A pair of (newly unused index, if @p k was found). The first element is only valid if the second is true.
   */
  template <typename ValueMover = DoNothingValueMover>
  std::pair<index_type, bool> erase(const key_type& k, ValueMover&& value_mover = {}) {
    auto [index, valid] = find(k);
    if (valid)
      index = erase_index(index, std::forward<ValueMover>(value_mover));
    return {index, valid};
  }

  /**
   * @brief Erase the key at @p index . May shuffle keys around to fill gaps.
   *
   * @return The newly unused index, after all shuffleing has completed.
   *
   * Every time a key is moved, calls @p value_mover with the @c (new,old) indexes,
   *   so that an external array of data be kept synchronized.
   * Also, a caller may consider destructing or invalidating the element of an external array that
   *   corresponds to the returned index.
   */
  template <typename ValueMover = DoNothingValueMover>
  index_type erase_index(index_type index, ValueMover&& value_mover = {}) {
    FWASSERT("erasing invalid index", index >= 0 && index < (index_type)max_entries_);
    FWASSERT("erasing invalid key", key_is_valid[index]);
    auto old_key = std::move(keys[index]);
    (void)old_key;  // drop it off the stack -- just call destructor
    key_is_valid[index] = false;
    auto last_deleted_index = index;
    for (auto actual_index = index + 1;; ++actual_index) {
      if (std::size_t(actual_index) == max_entries_) {
        actual_index = 0;
      }
      if (not key_is_valid[actual_index]) {  // sufficient to prevent infinite loops!
        break;
      }
      const auto hash = hasher_instance(keys[actual_index]);
      const index_type natural_index = hash % max_entries_;

      const bool natural_index_requires_loop_around = actual_index < natural_index;
      const bool looped_around_since_last_delete = actual_index < last_deleted_index;

      bool can_move_back = false;
      if (natural_index_requires_loop_around) {
        if (looped_around_since_last_delete) {
          // ai < ni && ai < ldi
          // eg. [ai,x,x,x,ni,ldi] (yes)
          // eg. [ai,x,x,x,x,ni=ldi] (yes)
          // eg. [ai,x,x,x,ldi,ni] (no)
          can_move_back = natural_index <= last_deleted_index;
        } else {
          // ldi < ai < ni    (ldi == ai is always impossible)
          // eg. [ldi,v,ai,x,ni] (yes, only possible ordering)
          can_move_back = true;
        }
      } else {
        if (looped_around_since_last_delete) {
          // ni <= ai && ai < ldi
          // eg. [ni,v,v,v,ai,ldi] (no)
          // eg. [v,v,v,v,ni=ai,ldi] (no)
          can_move_back = false;
        } else {
          // ni <= ai && ldi < ai    (ldi == ai is always impossible)
          // eg. [ni,ldi,v,v,v,ai] (yes)
          // eg. [ni=ldi,v,v,v,v,ai] (yes)
          // eg. [ldi,v,v,v,v,ni=ai] (no)
          // eg. [ldi,ni,v,v,v,ai] (no)
          can_move_back = natural_index <= last_deleted_index;
        }
      }

      if (can_move_back) {
        // swap it back
        key_is_valid[actual_index] = false;
        key_is_valid[last_deleted_index] = true;
        keys[last_deleted_index] = std::move(keys[actual_index]);
        std::forward<ValueMover>(value_mover)(last_deleted_index, actual_index);
        last_deleted_index = actual_index;  // remember this place
      }
    }
    return last_deleted_index;
  }

  void clear() { key_is_valid.reset(); }

 private:
  KeyStorage keys;
  std::bitset<max_entries_> key_is_valid;

  // Note: this and the key_equal_instance each take up one byte, even if empty.
  // Solutions are c++20's [[no_unique_address]] or the empty base optimization.
  hasher hasher_instance;
  key_equal key_equal_instance;
};

/**
 * @brief A hash-based mapping from a key to a value, with a compile-time fixed maximum number of entries
 *
 * The current implementation is separate key (via. StaticHashedKeyTable) and value arrays,
 *   as to not loose memory efficiency to padding.
 * Additionally, StaticHashedKeyTable uses linear probing.
 *
 * The interface is a bit different from @c std::unordered_map, and the guarantees are different.
 * In particular, references to keys and mapped types *may be invalidated* when erasing unrelated keys.
 * Also, since this hash table never automatically rebalances, references are never otherwise invalidated.
 *
 * Regarding the interface, the main difference is that where @c unordered_map uses @c pair<const_key,mapped_type>,
 *   this hash table uses just @c mapped_type .
 * This also extends to functions that typically take or return iterators --
 *   this hash table uses just @c mapped_type* .
 *
 * Known shortcomings include requiring @p Key and @p MappedType to be default constructable
 *   and that key and value (con/de)structors are called no matter what in the (con/de)structor.
 * This is generally not a huge problem, as most types used in FW are trivially (con/de)structable.
 *
 * Future improvements could include supporting supporting an external value array,
 *   which would allow putting the values in L1, and keeping the key array in the data RAM.
 */
template <
    typename Key,
    typename MappedType,
    std::size_t max_entries_,
    typename MappedTypeStorage = std::array<MappedType, max_entries_>,
    typename KeyStorage = std::array<Key, max_entries_>,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<>>
class StaticHashTable {
 public:
  using key_type = Key;
  using mapped_type = MappedType;
  using hasher = Hash;
  using key_equal = KeyEqual;

  StaticHashTable() : StaticHashTable(KeyStorage()) {}  // don't want an explict default constructor.
  explicit StaticHashTable(
      KeyStorage key_storage,
      MappedTypeStorage data_storage = MappedTypeStorage(),
      hasher h = hasher(),
      key_equal ke = key_equal()) :
      keys(std::move(key_storage), std::move(h), std::move(ke)), values(std::move(data_storage)) {
    FWASSERT("can't use nullptr for data storage", values.data() != nullptr);
  }

  /** @brief Search for the key @p k , and return the associated value. If not found, returns @c nullptr . */
  const mapped_type* find(const key_type& k) const {
    const auto [index, valid] = keys.find(k);
    return valid ? &values[index] : nullptr;
  }

  /** @brief non-const version of the above */
  mapped_type* find(const key_type& k) {
    const auto [index, valid] = keys.find(k);
    return valid ? &values[index] : nullptr;
  }

  /** @brief If k is known to be in this hash table, return directly by reference. @c FWASSERT s if not found. */
  const mapped_type& at(const key_type& k) const {
    auto result_ptr = find(k);
    FWASSERT("Could not find entry in a hash table", result_ptr != nullptr);
    return *result_ptr;
  }

  /** @brief non-const version of the above */
  mapped_type& at(const key_type& k) {
    auto result_ptr = find(k);
    FWASSERT("Could not find entry in a hash table", result_ptr != nullptr);
    return *result_ptr;
  }

  /**
   * @brief Add a key and value to this table. Does not overwrite.
   *
   * Does not invalidate any references.
   * @return The inserted value and true, or the existing value and false.
   */
  std::pair<mapped_type*, bool> insert(key_type k, mapped_type v) {
    const auto [index, new_key] = keys.insert(k);
    if (new_key)
      values[index] = std::move(v);
    return {&values[index], new_key};
  }

  /**
   * @brief Remove a value by key
   *
   * May invalidate references.
   * @return Did the key exist.
   */
  bool erase(const key_type& k) { return keys.erase(k, make_value_mover()).second; }

  /**
   * @brief Remove a value by value pointer/iterator.
   *
   * May invalidate references.
   * Behaviour is undefined if @p iter is not a non-null result of calling @c find .
   * Useful for efficiently replacing entries. Eg. @c ht.erase(ht.find(key)) is maximally efficient.
   */
  void erase_iter(const mapped_type* iter) { keys.erase_index(iter - values.data(), make_value_mover()); }

  /** @brief Erase by key, and return the mapped value that was there. */
  mapped_type extract(const key_type& k) {
    const auto [index, valid] = keys.erase(k);
    FWASSERT("Cannot extract non-existant key", valid);
    (void)valid;
    return std::move(values[index]);
  }

  /** @brief Reset to have no entries */
  void clear() { keys.clear(); }

 private:
  auto make_value_mover() {
    return [this](auto new_index, auto old_index) { values[new_index] = std::move(values[old_index]); };
  }

  StaticHashedKeyTable<key_type, max_entries_, KeyStorage, hasher, key_equal> keys;
  MappedTypeStorage values;
};

/**
 * @brief Simple adapter from a pointer to something more like @c std::array .
 *
 * Can serve as the @c MappedTypeStorage template parameter for StaticHashTable .
 */
template <typename V>
struct ExternalArray {
  ExternalArray(V* base) : base(base) {}

  V* base;

  V* data() {
#ifdef TENSIX_FIRMWARE
    return base;
#else
    return core->fw_ptr_to_emule(base);
#endif
  }
  const V* data() const {
#ifdef TENSIX_FIRMWARE
    return base;
#else
    return core->fw_ptr_to_emule(base);
#endif
  }
  V& operator[](std::size_t i) { return data()[i]; }
  const V& operator[](std::size_t i) const { return data()[i]; }
};
