(function () {
  "use strict";

  let theme = "light";
  try {
    const saved = window.localStorage.getItem("hal-docs-theme");
    if (saved === "light" || saved === "dark") {
      theme = saved;
    } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
      theme = "dark";
    }
  } catch (error) {
    if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) theme = "dark";
  }
  document.documentElement.dataset.theme = theme;
}());
