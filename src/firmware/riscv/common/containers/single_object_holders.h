#pragma once

/**
 * @brief Like unique_ptr, but in-place.
 * 
 * Note: not tested or used, just an idea.
 */
template <typename T>
class InitializeLater {
public:
  static constexpr auto size_in_bytes = sizeof(T);
  static constexpr auto alignment = alignof(T);

  InitializeLater() = default;
  InitializeLater(T t) { initialize(std::move(t)); }
  InitializeLater(const T& t) { initialize(t); }
  InitializeLater(InitializeLater src) {
    if (not src.is_initialized()) return;
    initialize(std::move(*src));
  }
  InitializeLater(const InitializeLater& src) {
    if (not src.is_initialized()) return;
    initialize(*src);
  }

  ~InitializeLater() {
    if (is_initialized()) {
      deinitialize();
    }
  }

  template <typename... Args>
  void reset(Args&&... args) {
    if (is_initialized()) deinitialize();
    initialize(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void initialize(Args&&... args) {
    FWASSERT_NSE("this InitializeLater is already initialized", not is_initialized());
    new (&data) T(std::forward<Args>(args)...);  // placement new (does not call malloc)
  }

  void deinitialize() {
    assert_init();
    this->get()->~T();  // manually call the destructor
  }

  T* get() {
    assert_init();
    return (T*)(&data);
  }

  const T* get() const {
    assert_init();
    return (const T*)(&data);
  }

  T& operator*() { return *get(); }
  T* operator->() { return get(); }
  const T& operator*() const { return *get(); }
  const T* operator->() const { return get(); }

  void assert_init() const {
    FWASSERT_NSE("this InitializeLater is not initialized", is_initialized());
  }

  bool is_initialized() const { return is_initialized_; }

private:
  alignas(alignment) unsigned char data[size_in_bytes];
  bool is_initialized_ = false;
};
