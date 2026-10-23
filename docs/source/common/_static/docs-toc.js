(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', function () {
    buildPageToc();
  });

  function buildPageToc() {
    // Skip the home/landing page
    if (document.querySelector('.hero')) return;

    var content = document.querySelector('.rst-content .document') ||
                  document.querySelector('.rst-content');
    if (!content) return;

    var headings = content.querySelectorAll('h2, h3');
    if (headings.length < 2) return;

    var toc = document.createElement('nav');
    toc.className = 'tt-page-toc';
    toc.setAttribute('aria-label', 'Page contents');

    var ul = document.createElement('ul');
    var items = [];

    headings.forEach(function (h) {
      var id = anchorId(h);
      if (!id) return; /* nothing to link to; skip rather than invent an anchor */

      /* Percent-encode the id before it reaches an href. The value is already
       * a same-page fragment because of the leading '#', so no scheme can be
       * introduced; encoding additionally guarantees the whole href stays
       * within the URL fragment grammar whatever Sphinx put in the id. */
      var fragment = '#' + encodeURIComponent(id);

      var li = document.createElement('li');
      li.className = h.tagName === 'H3' ? 'toc-h3' : 'toc-h2';

      var a = document.createElement('a');
      a.setAttribute('href', fragment);

      /* Strip headerlink anchors (¶ / [] symbols) before reading text */
      var clone = h.cloneNode(true);
      clone.querySelectorAll('a.headerlink').forEach(function (el) { el.remove(); });
      a.textContent = clone.textContent.trim();

      a.addEventListener('click', function (e) {
        e.preventDefault();
        h.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.pushState(null, '', fragment);
        setActive(a);
      });

      li.appendChild(a);
      ul.appendChild(li);
      items.push({ heading: h, link: a });
    });

    if (!items.length) return;

    toc.appendChild(ul);
    document.body.appendChild(toc);

    setupScrollSpy(items);
  }

  /* Sphinx puts the anchor on the enclosing <section>, not on the heading, and
   * dedupes repeated titles there ("parameters", then "id1"). So read the id
   * that already exists instead of deriving one from the heading text: a
   * derived id would collide with the section's own id, and two sections with
   * the same title would derive the same value and produce two TOC entries
   * pointing at the first one. Older Sphinx used <div class="section" id>. */
  function anchorId(h) {
    if (h.id) return h.id;

    var node = h.parentNode;
    while (node && node.nodeType === 1) {
      if (node.tagName === 'SECTION' || node.classList.contains('section')) {
        if (node.id) return node.id;
      }
      node = node.parentNode;
    }

    /* Sphinx's own permalink already points at the right anchor. */
    var headerlink = h.querySelector('a.headerlink');
    var href = headerlink && headerlink.getAttribute('href');
    if (href && href.charAt(0) === '#') return decodeURIComponent(href.slice(1));

    return '';
  }

  function setActive(activeLink) {
    var toc = document.querySelector('.tt-page-toc');
    if (!toc) return;
    toc.querySelectorAll('a').forEach(function (a) {
      a.classList.remove('active');
    });
    activeLink.classList.add('active');
  }

  function setupScrollSpy(items) {
    var navbarHeight = parseInt(
      getComputedStyle(document.documentElement)
        .getPropertyValue('--tt-navbar-height') || '72', 10
    );

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          /* Match on the heading node itself: two sections may legitimately
           * share a title, so the label is not a key and the id lives on the
           * section rather than on the element being observed. */
          items.forEach(function (item) {
            item.link.classList.toggle('active', item.heading === entry.target);
          });
        }
      });
    }, {
      rootMargin: '-' + (navbarHeight + 8) + 'px 0px -70% 0px',
      threshold: 0
    });

    items.forEach(function (item) {
      observer.observe(item.heading);
    });

    items[0].link.classList.add('active');
  }
})();
